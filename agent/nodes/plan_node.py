"""Master planning node that analyzes intent and emits structured tool calls.

Two planner paths share one output contract:
- LLM planner (planner_v2): decomposes via a chat model, JSON-validated.
- Deterministic fallback (decomposer rules): used when no LLM is configured,
  the call fails, or the JSON doesn't validate — so evaluation and CI stay
  runnable offline.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from agent.llm import invoke_llm, is_llm_enabled, llm_usage_entry
from agent.nodes.evidence_math import Operand, resolve_arithmetic
from agent.prompts import load_prompt
from agent.state.schema import AgentStateV1
from agent.tools.decomposer import SubTask, decompose_query
from agent.tools.graph_tool import GraphQueryInput
from agent.tools.safe_math import SafeMathInput
from agent.tools.table_tool import TableExtractInput
from agent.tools.vector_tool import VectorSearchInput

GRAPH_VERSION = "v1.0.0"

_PLANNER_PROMPT_NAME = "planner_v2"


_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class LLMPlan(BaseModel):
    """Strict shape the LLM planner must emit (mirrors DecompositionOutput)."""

    is_multi_hop: bool = False
    reasoning_plan: str = ""
    sub_tasks: list[SubTask] = Field(default_factory=list, max_length=16)


def validate_tool_payload(tool: str, payload: dict[str, Any]) -> None:
    schemas: dict[str, type[BaseModel]] = {
        "graph_retrieval": GraphQueryInput,
        "vector_retrieval": VectorSearchInput,
        "safe_math": SafeMathInput,
        "table_extract": TableExtractInput,
    }
    schema = schemas[tool]
    if set(payload) - set(schema.model_fields):
        raise ValueError("Unknown tool parameters")
    schema.model_validate(payload, strict=True)
    if any(isinstance(value, str) and not value.strip() for value in payload.values()):
        raise ValueError("Empty tool parameter")


def _extract_json(content: str) -> dict[str, Any] | None:
    """Parses the first JSON object from an LLM response, fences tolerated."""
    text = str(content).strip()
    fence = _JSON_FENCE_RE.match(text)
    if fence:
        text = fence.group(1)

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _validate_llm_plan(plan: LLMPlan) -> list[str]:
    """Returns a list of contract violations (empty list = valid)."""
    problems: list[str] = []
    seen: set[str] = set()
    for i, task in enumerate(plan.sub_tasks, start=1):
        expected = f"task_{i}"
        if task.task_id != expected:
            problems.append(
                f"task_id {task.task_id} out of sequence (expected {expected})"
            )
        bad_deps = [d for d in task.dependencies if d not in seen]
        if bad_deps:
            problems.append(
                f"{task.task_id} depends on unknown/forward tasks: {bad_deps}"
            )
        if not isinstance(task.query_payload, dict) or not task.query_payload:
            problems.append(f"{task.task_id} has empty payload")
        try:
            validate_tool_payload(task.target_tool, task.query_payload)
            if task.target_tool == "safe_math":

                def reference(node: ast.expr) -> Operand | None:
                    if (
                        isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)
                        and node.attr == "amount"
                        and node.value.id in task.dependencies
                    ):
                        return Operand("1.0", [], "number", 1)
                    return None

                resolve_arithmetic(task.query_payload["expression"], reference)
        except (ValueError, KeyError, SyntaxError) as exc:
            problems.append(f"{task.task_id} invalid {task.target_tool} payload: {exc}")
        seen.add(task.task_id)

    has_retrieval = any(
        t.target_tool in {"graph_retrieval", "vector_retrieval"} for t in plan.sub_tasks
    )
    has_math = any(t.target_tool == "safe_math" for t in plan.sub_tasks)
    if has_math and not has_retrieval:
        problems.append(
            "safe_math scheduled without any retrieval task to source values from"
        )
    return problems


def _plan_with_llm(
    state: AgentStateV1,
    usage: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str] | None:
    """Attempts LLM decomposition. Returns (pending_calls, log, usage)."""
    if not is_llm_enabled():
        return None

    _, prompt_version, prompt_hash = load_prompt(_PLANNER_PROMPT_NAME)
    company = state.company_identifier or "UNKNOWN"
    user_prompt = f"Company identifier: {company}\n" f"User question: {state.input}"
    if state.iteration_count:
        last_tasks = [
            (
                c.get("target_tool"),
                c.get("payload"),
                (state.sub_task_results.get(c["task_id"]) or {}).get("error"),
            )
            for c in state.tool_calls
        ]
        user_prompt += (
            "\nPrevious plan produced insufficient evidence. Revise the retrievals:\n"
            + json.dumps(last_tasks)
        )

    result = invoke_llm(
        system_prompt=load_prompt(_PLANNER_PROMPT_NAME)[0],
        user_prompt=user_prompt,
        max_tokens=800,
        task="planner",
    )
    usage.append(llm_usage_entry("planner", result))
    if result is None:
        return None

    parsed = _extract_json(result.content)
    if parsed is None:
        return None

    try:
        plan = LLMPlan(**parsed)
    except ValidationError:
        return None

    problems = _validate_llm_plan(plan)
    if problems:
        print(
            f"[planner] LLM plan rejected ({'; '.join(problems[:3])}); "
            f"falling back to deterministic decomposition.",
            flush=True,
        )
        return None

    if not plan.sub_tasks:

        return None

    pending_calls: list[dict[str, Any]] = []
    for task in plan.sub_tasks:
        payload = dict(task.query_payload)
        if task.target_tool in {"graph_retrieval", "vector_retrieval"}:
            for scoped_key in ("record_id", "record_ids", "split"):
                payload.pop(scoped_key, None)
        pending_calls.append(
            {
                "task_id": task.task_id,
                "target_tool": task.target_tool,
                "payload": payload,
                "dependencies": task.dependencies,
                "status": "PENDING",
                "verify_formula": task.target_tool == "safe_math",
            }
        )

    log = (
        f"[Planner {prompt_version} (hash:{prompt_hash}) llm:{result.model}] "
        f"Generated plan with {len(pending_calls)} tasks. Strategy: {plan.reasoning_plan}. "
        f"LLM tokens: {result.prompt_tokens}+{result.completion_tokens}."
    )
    return pending_calls, log


def rekey_calls(calls: list[dict[str, Any]], generation: int) -> list[dict[str, Any]]:
    names = {
        c["task_id"]: f"task_{100 * generation + i}" for i, c in enumerate(calls, 1)
    }
    updated = []
    for original in calls:
        payload = dict(original["payload"])
        if original["target_tool"] == "safe_math":
            payload["expression"] = re.sub(
                r"\btask_\d+(?=\.amount\b)",
                lambda m: names[m[0]],
                payload["expression"],
            )
        updated.append(
            {
                **original,
                "task_id": names[original["task_id"]],
                "payload": payload,
                "dependencies": [names[d] for d in original.get("dependencies", [])],
                "status": "PENDING",
            }
        )
    return updated


def plan_node(state: AgentStateV1) -> dict[str, Any]:
    """Evaluates question and scratchpad to construct or refine tool execution queue."""
    _, prompt_version, prompt_hash = load_prompt("planner_v1")
    new_iteration = state.iteration_count + 1

    usage: list[dict[str, Any]] = []
    llm_outcome = _plan_with_llm(state, usage)
    if llm_outcome is not None:
        pending_calls, log_entry = llm_outcome
    else:

        company = state.company_identifier or "UNKNOWN"
        decomp = decompose_query(query=state.input, company=company)

        pending_calls = []
        for task in decomp.sub_tasks:
            pending_calls.append(
                {
                    "task_id": task.task_id,
                    "target_tool": task.target_tool,
                    "payload": task.query_payload,
                    "dependencies": task.dependencies,
                    "status": "PENDING",
                }
            )

        log_entry = (
            f"[Planner {prompt_version}@{GRAPH_VERSION} (hash:{prompt_hash}) rules] "
            f"Generated plan with {len(pending_calls)} tasks. Strategy: {decomp.reasoning_plan}"
        )
        if state.iteration_count:
            for call in pending_calls:
                if call["target_tool"] == "vector_retrieval":
                    call["payload"]["top_k"] = min(20, 5 * new_iteration)

    if state.company_identifier:
        for call in pending_calls:
            if call["target_tool"] in {"graph_retrieval", "vector_retrieval"}:
                call["payload"]["company_identifier"] = state.company_identifier

    if state.record_id:
        for call in pending_calls:
            if call.get("target_tool") in {"graph_retrieval", "vector_retrieval"}:
                call["payload"] = {
                    **call.get("payload", {}),
                    "record_id": state.record_id,
                }

        existing_ids = {c.get("task_id") for c in pending_calls}
        table_task_id = "task_table"
        n = 1
        while table_task_id in existing_ids:
            n += 1
            table_task_id = f"task_table_{n}"
        pending_calls.append(
            {
                "task_id": table_task_id,
                "target_tool": "table_extract",
                "payload": {"record_id": state.record_id},
                "dependencies": [],
                "status": "PENDING",
            }
        )
        log_entry = f"{log_entry} + full-table extraction ({table_task_id})."

    if state.iteration_count:
        pending_calls = rekey_calls(pending_calls, state.iteration_count)
    return {
        "iteration_count": new_iteration,
        "tool_calls": pending_calls,
        "scratchpad": [log_entry],
        "llm_usage": usage,
        "fused_context": "",
        "evidence": [],
        "citations": [],
        "risk_evaluated": False,
    }
