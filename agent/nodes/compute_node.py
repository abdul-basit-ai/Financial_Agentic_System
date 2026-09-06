"""Deterministic arithmetic node resolving parameters and calling safe_math."""

from __future__ import annotations

import re
from typing import Any

from agent.state.schema import AgentStateV1
from agent.tools.safe_math import SafeMathInput, safe_math_tool


def _resolve_dynamic_expression(expression: str, tool_results: list[dict[str, Any]]) -> str:
    """Substitutes variables (e.g. task_1.amount) with concrete numbers from past tool results."""
    resolved = expression
    pattern = re.compile(r"\b(task_\d+)\.amount\b")
    matches = pattern.findall(expression)

    # Build lookup map: task_id -> numeric amount
    lookup: dict[str, float] = {}
    for res in tool_results:
        t_id = res.get("task_id")
        data = res.get("data")
        if t_id and data and isinstance(data, dict):
            # If from graph retrieval
            records = data.get("records", [])
            if records and records[0].get("amount") is not None:
                lookup[t_id] = float(records[0]["amount"])
            # If from previous math calculation
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
    scratchpad_logs: list[str] = []

    for call in state.tool_calls:
        if call.get("target_tool") == "safe_math" and call.get("status") == "PENDING":
            raw_expr = call.get("payload", {}).get("expression", "")
            concrete_expr = _resolve_dynamic_expression(raw_expr, state.tool_results)

            result = safe_math_tool(SafeMathInput(expression=concrete_expr))
            executed_results.append({
                "task_id": call.get("task_id"),
                "tool_name": "safe_math",
                "success": result.success,
                "data": result.data.model_dump() if result.data else None,
                "error": result.error,
            })
            val = result.data.formatted if result.data else "ERROR"
            scratchpad_logs.append(f"[Safe Math] Evaluated '{concrete_expr}' -> Result: {val}")
        else:
            remaining_calls.append(dict(call))

    return {
        "tool_calls": remaining_calls,
        "tool_results": executed_results,
        "scratchpad": scratchpad_logs,
    }