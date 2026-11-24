"""Resolve planned arithmetic against current retrievals and earlier calculations."""

from __future__ import annotations

import ast
import json
from typing import Any

from agent.llm import invoke_llm, llm_usage_entry
from agent.nodes.evidence_math import Operand, calculation_data, resolve_arithmetic
from agent.nodes.plan_node import _extract_json
from agent.nodes.relevance import best_matching_record, question_terms
from agent.prompts import load_prompt
from agent.state.schema import AgentStateV1


def select_record(
    records: list[dict[str, Any]],
    terms: set[str],
    year: int | None = None,
    metric: str | None = None,
) -> dict[str, Any] | None:
    if year is not None:
        records = [
            r
            for r in records
            if str(year)
            in {
                str(r.get("year")),
                str(r.get("column_header", "")).strip(),
                str(r.get("row_label", "")).strip(),
            }
        ]
    required = question_terms(metric or "") or terms
    match = best_matching_record(records, required)
    if match is None:
        return None
    labels = question_terms(
        f"{match.get('row_label', '')} {match.get('column_header', '')}"
    )
    if metric and not required.issubset(labels):
        return None
    return match


def compute_node(state: AgentStateV1) -> dict[str, Any]:
    calls = {call["task_id"]: call for call in state.tool_calls}
    completed = {r["task_id"]: r for r in state.tool_results if r.get("task_id")}
    completed.update(state.sub_task_results)
    updates: dict[str, dict[str, Any]] = {}
    remaining: list[dict[str, Any]] = []
    logs: list[str] = []
    usage: list[dict[str, Any]] = []
    for task_id, call in calls.items():
        if call.get("target_tool") != "safe_math" or call.get("status") != "PENDING":
            remaining.append(call)
            continue
        dependencies = set(call.get("dependencies", []))

        def source(node: ast.expr) -> Operand | None:
            if not isinstance(node, ast.Attribute) or not isinstance(
                node.value, ast.Name
            ):
                return None
            ref = node.value.id
            if node.attr != "amount" or ref not in dependencies:
                raise ValueError("Undeclared calculation dependency")
            env = completed.get(ref) or {}
            data = env.get("data") or {}
            if not env.get("success"):
                raise ValueError("Missing successful dependency")
            if env.get("tool_name") == "safe_math" and data.get("citations"):
                return Operand(
                    repr(float(data["result"])),
                    data["citations"],
                    data["unit_label"],
                    data["scale_multiplier"],
                )
            payload = calls.get(ref, {}).get("payload", {})
            record = select_record(
                data.get("records", []),
                question_terms(state.input),
                payload.get("year"),
                payload.get("metric_name"),
            )
            if record is None:
                raise ValueError("Missing or ambiguous evidence for " + ref)
            return Operand.from_record(record)

        program = call.get("payload", {}).get("expression", "")
        envelope: dict[str, Any] = {"task_id": task_id, "tool_name": "safe_math"}
        try:
            operand = resolve_arithmetic(program, source)
            data = calculation_data(operand, program)
            if call.get("verify_formula"):
                proposal = {"question": state.input, "candidates": [data]}
                verdict = invoke_llm(
                    system_prompt=load_prompt("program_verify_v1")[0],
                    user_prompt=json.dumps(proposal),
                    task="verifier",
                    max_tokens=300,
                    temperature=0.0,
                )
                usage.append(llm_usage_entry("verifier", verdict))
                parsed = (_extract_json(verdict.content) if verdict else None) or {}
                selected = parsed.get("selected_index")
                if (
                    type(selected) is not int
                    or selected != 0
                    or parsed.get("corrected_expression") is not None
                ):
                    raise ValueError("Verifier did not confirm planned arithmetic")
            envelope.update(success=True, data=data, error=None)
            logs.append(f"[Safe Math] {data['expression']} = {data['formatted']}")
        except (ValueError, ArithmeticError, TypeError, SyntaxError) as exc:
            envelope.update(success=False, data=None, error=str(exc))
            logs.append(f"[Safe Math] {task_id} failed: {exc}")
        completed[task_id] = updates[task_id] = envelope
        remaining.append({**call, "status": "COMPLETED"})
    return {
        "tool_calls": remaining,
        "tool_results": list(updates.values()),
        "sub_task_results": updates,
        "scratchpad": logs,
        "risk_evaluated": False,
        "llm_usage": usage,
    }
