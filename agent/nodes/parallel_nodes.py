"""Parallel execution nodes for dynamic fan-out and fan-in using LangGraph Send API."""

from __future__ import annotations

from typing import Any

try:
    from langgraph.types import Send
except ImportError:
    try:
        from langgraph.constants import Send
    except ImportError:
        from langgraph.graph import Send

from agent.state.schema import AgentStateV1
from agent.tools.graph_tool import GraphQueryInput, graph_retrieval_tool
from agent.tools.safe_math import SafeMathInput, safe_math_tool
from agent.tools.vector_tool import VectorSearchInput, vector_retrieval_tool


def sub_task_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Isolated worker node executing a single SubTask dispatched via Send API."""
    task_id = payload.get("task_id", "task_unknown")
    target_tool = payload.get("target_tool")
    tool_payload = payload.get("payload", {})

    result_data: Any = None
    success = False
    error_msg: str | None = None
    log_entry = ""

    if target_tool == "graph_retrieval":
        query_input = GraphQueryInput(
            company_identifier=tool_payload.get("company_identifier", "UNKNOWN"),
            metric_name=tool_payload.get("metric_name"),
            year=tool_payload.get("year"),
            record_id=tool_payload.get("record_id"),
        )
        res = graph_retrieval_tool(query_input)
        success = res.success
        result_data = res.data.model_dump() if res.data else None
        error_msg = res.error
        count = res.data.total_found if res.data else 0
        log_entry = f"[Parallel Worker: {task_id}] Graph retrieved {count} records."

    elif target_tool == "vector_retrieval":
        query_input = VectorSearchInput(
            query_text=tool_payload.get("query_text", ""),
            top_k=tool_payload.get("top_k", 5),
        )
        res = vector_retrieval_tool(query_input)
        success = res.success
        result_data = res.data.model_dump() if res.data else None
        error_msg = res.error
        count = res.data.total_found if res.data else 0
        log_entry = f"[Parallel Worker: {task_id}] Vector retrieved {count} chunks."

    elif target_tool == "safe_math":
        raw_expr = tool_payload.get("expression", "0")
        res = safe_math_tool(SafeMathInput(expression=raw_expr))
        success = res.success
        result_data = res.data.model_dump() if res.data else None
        error_msg = res.error
        val = res.data.formatted if res.data else "ERROR"
        log_entry = f"[Parallel Worker: {task_id}] Math calculated '{raw_expr}' -> {val}"

    else:
        error_msg = f"Unsupported tool '{target_tool}' in sub_task_worker"
        log_entry = f"[Parallel Worker: {task_id}] Failed: {error_msg}"

    envelope = {
        "task_id": task_id,
        "tool_name": target_tool,
        "success": success,
        "data": result_data,
        "error": error_msg,
    }

    return {
        "sub_task_results": {task_id: envelope},
        "tool_results": [envelope],
        "scratchpad": [log_entry],
    }


def fan_out_router(state: AgentStateV1) -> list[Send] | str:
    """Evaluates pending sub-tasks and dynamically fans out independent branches via Send."""
    ready_tasks: list[dict[str, Any]] = []

    for call in state.tool_calls:
        if call.get("status") == "PENDING":
            deps = call.get("dependencies", [])
            # Ready if all dependencies exist in sub_task_results
            if not deps or all(d in state.sub_task_results for d in deps):
                # Only fan out graph and vector retrieval tasks
                if call.get("target_tool") in {"graph_retrieval", "vector_retrieval"}:
                    ready_tasks.append(call)

    if ready_tasks:
        return [
            Send("sub_task_worker", {
                "task_id": t.get("task_id"),
                "target_tool": t.get("target_tool"),
                "payload": t.get("payload", {}),
            })
            for t in ready_tasks
        ]

    # Check if a safe_math task is ready
    has_pending_math = any(
        c.get("target_tool") == "safe_math" and c.get("status") == "PENDING"
        for c in state.tool_calls
    )
    if has_pending_math:
        return "compute"

    # Default routing if no tasks to fan out
    if state.tool_results:
        return "fuse_context"

    return "synthesize_answer"


def aggregate_sub_tasks_node(state: AgentStateV1) -> dict[str, Any]:
    """Fan-in synchronization barrier: marks completed tasks and logs barrier convergence."""
    completed_task_ids = set(state.sub_task_results.keys())
    updated_calls = []

    for call in state.tool_calls:
        call_copy = dict(call)
        if call_copy.get("task_id") in completed_task_ids:
            call_copy["status"] = "COMPLETED"
        updated_calls.append(call_copy)

    log_entry = (
        f"[Parallel Fan-In Barrier] Converged {len(completed_task_ids)} completed tasks: "
        f"{sorted(list(completed_task_ids))}."
    )

    return {
        "tool_calls": updated_calls,
        "scratchpad": [log_entry],
    }