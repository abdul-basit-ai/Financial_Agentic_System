"""Deterministic arithmetic node resolving parameters from parallel sub-task results."""

from __future__ import annotations

import re
from typing import Any

from agent.state.schema import AgentStateV1
from agent.tools.safe_math import SafeMathInput, safe_math_tool


def _resolve_dynamic_expression(
    expression: str,
    tool_results: list[dict[str, Any]],
    sub_task_results: dict[str, Any] | None = None,
) -> str:
    """Substitutes dynamic task references (e.g., task_1.amount) with concrete values."""
    resolved = expression
    pattern = re.compile(r"\b(task_\d+)\.amount\b")
    matches = pattern.findall(expression)

    # Build lookup map: task_id -> numeric amount
    lookup: dict[str, float] = {}

    # Check parallel sub_task_results first
    if sub_task_results:
        for t_id, envelope in sub_task_results.items():
            data = envelope.get("data")
            if isinstance(data, dict):
                records = data.get("records", [])
                if records and records[0].get("amount") is not None:
                    lookup[t_id] = float(records[0]["amount"])
                elif "result" in data:
                    lookup[t_id] = float(data["result"])

    # Supplement with sequential tool_results
    for res in tool_results:
        t_id = res.get("task_id")
        data = res.get("data")
        if t_id and data and isinstance(data, dict) and t_id not in lookup:
            records = data.get("records", [])
            if records and records[0].get("amount") is not None:
                lookup[t_id] = float(records[0]["amount"])
            elif "result" in data:
                lookup[t_id] = float(data["result"])

    for m in matches:
        if m in lookup:
            resolved = resolved.replace(f"{m}.amount", str(lookup[m]))

    return resolved


def compute_node(state: AgentStateV1) -> dict[str, Any]:
    """Executes safe arithmetic calculations using AST evaluator."""
    remaining_calls: list[dict[str, Any]] = []
    executed_results: list[dict[str, Any]] = []
    sub_task_updates: dict[str, Any] = {}
    scratchpad_logs: list[str] = []

    for call in state.tool_calls:
        if call.get("target_tool") == "safe_math" and call.get("status") == "PENDING":
            raw_expr = call.get("payload", {}).get("expression", "")
            concrete_expr = _resolve_dynamic_expression(
                raw_expr, state.tool_results, state.sub_task_results
            )

            result = safe_math_tool(SafeMathInput(expression=concrete_expr))
            task_id = call.get("task_id", "task_math")

            envelope = {
                "task_id": task_id,
                "tool_name": "safe_math",
                "success": result.success,
                "data": result.data.model_dump() if result.data else None,
                "error": result.error,
            }
            executed_results.append(envelope)
            sub_task_updates[task_id] = envelope

            val = result.data.formatted if result.data else "ERROR"
            scratchpad_logs.append(f"[Safe Math] Evaluated '{concrete_expr}' -> Result: {val}")
        else:
            remaining_calls.append(dict(call))

    return {
        "tool_calls": remaining_calls,
        "tool_results": executed_results,
        "sub_task_results": sub_task_updates,
        "scratchpad": scratchpad_logs,
    }