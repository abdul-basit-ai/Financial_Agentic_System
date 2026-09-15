"""Master planning node that analyzes intent and emits structured tool calls.

Two planner paths share one output contract:
- LLM planner (planner_v2): decomposes via a chat model, JSON-validated.
- Deterministic fallback (decomposer rules): used when no LLM is configured,
  the call fails, or the JSON doesn't validate — so evaluation and CI stay
  runnable offline.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from agent.llm import invoke_llm, is_llm_enabled
from agent.prompts import load_prompt
from agent.state.schema import AgentStateV1
from agent.tools.decomposer import SubTask, decompose_query

GRAPH_VERSION = "v1.0.0"

_PLANNER_PROMPT_NAME = "planner_v2"

# Strips markdown code fences the model may add despite instructions
_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class LLMPlan(BaseModel):
    """Strict shape the LLM planner must emit (mirrors DecompositionOutput)."""

    is_multi_hop: bool = False
    reasoning_plan: str = ""
    sub_tasks: list[SubTask] = Field(default_factory=list)


def _extract_json(content: str) -> dict[str, Any] | None:
    """Parses the first JSON object from an LLM response, fences tolerated."""
    text = str(content).strip()
    fence = _JSON_FENCE_RE.match(text)
    if fence:
        text = fence.group(1)
    # Tolerate prose before/after the object
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
            problems.append(f"task_id {task.task_id} out of sequence (expected {expected})")
        bad_deps = [d for d in task.dependencies if d not in seen]
        if bad_deps:
            problems.append(f"{task.task_id} depends on unknown/forward tasks: {bad_deps}")
        if not isinstance(task.query_payload, dict) or not task.query_payload:
            problems.append(f"{task.task_id} has empty payload")
        seen.add(task.task_id)

    has_retrieval = any(t.target_tool in {"graph_retrieval", "vector_retrieval"} for t in plan.sub_tasks)
    has_math = any(t.target_tool == "safe_math" for t in plan.sub_tasks)
    if has_math and not has_retrieval:
        problems.append("safe_math scheduled without any retrieval task to source values from")
    return problems


def _plan_with_llm(state: AgentStateV1) -> tuple[list[dict[str, Any]], str] | None:
    """Attempts LLM decomposition. Returns (pending_calls, log) or None."""
    if not is_llm_enabled():
        return None

    _, prompt_version, prompt_hash = load_prompt(_PLANNER_PROMPT_NAME)
    company = state.company_identifier or "UNKNOWN"
    user_prompt = (
        f"Company identifier: {company}\n"
        f"User question: {state.input}"
    )

    result = invoke_llm(
        system_prompt=load_prompt(_PLANNER_PROMPT_NAME)[0],
        user_prompt=user_prompt,
        max_tokens=800,
    )
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
        # Model explicitly judged the question unanswerable; an empty plan
        # would stall the graph, so fall through to the deterministic planner
        # which always schedules retrieval.
        return None

    pending_calls: list[dict[str, Any]] = []
    for task in plan.sub_tasks:
        pending_calls.append({
            "task_id": task.task_id,
            "target_tool": task.target_tool,
            "payload": task.query_payload,
            "dependencies": task.dependencies,
            "status": "PENDING",
        })

    log = (
        f"[Planner {prompt_version} (hash:{prompt_hash}) llm:{result.model}] "
        f"Generated plan with {len(pending_calls)} tasks. Strategy: {plan.reasoning_plan}. "
        f"LLM tokens: {result.prompt_tokens}+{result.completion_tokens}, "
        f"est. cost ${result.cost_usd:.6f}."
    )
    return pending_calls, log


def plan_node(state: AgentStateV1) -> dict[str, Any]:
    """Evaluates question and scratchpad to construct or refine tool execution queue."""
    _, prompt_version, prompt_hash = load_prompt("planner_v1")
    new_iteration = state.iteration_count + 1

    # Check if this is an iterative refinement loop
    if state.tool_results and not state.final_answer:
        log_entry = (
            f"[Planner {prompt_version}@{GRAPH_VERSION} (hash:{prompt_hash})] "
            f"Iteration {new_iteration}: Reviewing {len(state.tool_results)} completed tool results."
        )
        return {
            "iteration_count": new_iteration,
            "scratchpad": [log_entry],
            "risk_evaluated": False,
        }

    llm_outcome = _plan_with_llm(state)
    if llm_outcome is not None:
        pending_calls, log_entry = llm_outcome
    else:
        # Macro-decomposition of user financial question (rule-based fallback)
        company = state.company_identifier or "UNKNOWN"
        decomp = decompose_query(query=state.input, company=company)

        pending_calls = []
        for task in decomp.sub_tasks:
            pending_calls.append({
                "task_id": task.task_id,
                "target_tool": task.target_tool,
                "payload": task.query_payload,
                "dependencies": task.dependencies,
                "status": "PENDING",
            })

        log_entry = (
            f"[Planner {prompt_version}@{GRAPH_VERSION} (hash:{prompt_hash}) rules] "
            f"Generated plan with {len(pending_calls)} tasks. Strategy: {decomp.reasoning_plan}"
        )

    return {
        "iteration_count": new_iteration,
        "tool_calls": pending_calls,
        "scratchpad": [log_entry],
        "risk_evaluated": False,
    }
