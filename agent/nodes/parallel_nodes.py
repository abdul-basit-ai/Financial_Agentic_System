"""Parallel execution nodes for dynamic fan-out and fan-in using LangGraph Send API."""

from __future__ import annotations

from typing import Any

from langgraph.types import Send
from pydantic import ValidationError
from typing_extensions import TypedDict

from agent.nodes.concurrency import ToolSlot
from agent.state.schema import AgentStateV1
from agent.tools.graph_tool import GraphQueryInput, graph_retrieval_tool
from agent.tools.safe_math import SafeMathInput, safe_math_tool
from agent.tools.table_tool import TableExtractInput, table_extract_tool
from agent.tools.vector_tool import VectorSearchInput, vector_retrieval_tool


class WorkerPayload(TypedDict):
    task_id: str
    target_tool: str
    payload: dict[str, Any]


def sub_task_worker(state: WorkerPayload) -> dict[str, Any]:
    payload = state
    try:
        return _execute_worker(dict(payload))
    except (ValidationError, TypeError, ValueError, KeyError) as exc:
        task_id = payload.get("task_id", "task_unknown")
        envelope = {
            "task_id": task_id,
            "tool_name": payload.get("target_tool"),
            "success": False,
            "data": None,
            "error": f"Invalid tool input: {type(exc).__name__}",
        }
        return {
            "tool_results": [envelope],
            "sub_task_results": {task_id: envelope},
            "scratchpad": [f"[Parallel Worker: {task_id}] Invalid tool input"],
        }


def _execute_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Isolated worker node executing a single SubTask dispatched via Send API.

    Concurrency is bounded per-tool via the ToolSlot semaphore registry
    (agent.nodes.concurrency) — parallel Send branches queue rather than
    overwhelming downstream stores; wait/contention is logged for Phase 12.
    """
    task_id = payload["task_id"]
    tool_name = payload["target_tool"]
    arguments = payload["payload"]
    envelope: dict[str, Any] = {
        "task_id": task_id,
        "tool_name": tool_name,
        "success": False,
        "data": None,
        "error": None,
    }
    with ToolSlot(tool_name) as slot:
        if slot.status == "timeout":
            envelope["error"] = f"Concurrency limit timeout for '{tool_name}'"
        else:
            result: Any
            if tool_name == "graph_retrieval":
                result = graph_retrieval_tool(
                    GraphQueryInput.model_validate(arguments, strict=True)
                )
            elif tool_name == "vector_retrieval":
                result = vector_retrieval_tool(
                    VectorSearchInput.model_validate(arguments, strict=True)
                )
            elif tool_name == "table_extract":
                result = table_extract_tool(
                    TableExtractInput.model_validate(arguments, strict=True)
                )
            elif tool_name == "safe_math":
                result = safe_math_tool(
                    SafeMathInput.model_validate(arguments, strict=True)
                )
            else:
                raise ValueError("Unsupported tool")
            envelope.update(
                success=result.success,
                data=result.data.model_dump() if result.data else None,
                error=result.error,
                metrics={
                    **result.metrics.model_dump(),
                    "wait_seconds": slot.wait_seconds,
                },
            )
    log = envelope["error"] or "Completed"
    return {
        "sub_task_results": {task_id: envelope},
        "tool_results": [envelope],
        "scratchpad": [f"[Parallel Worker: {task_id}] {tool_name}: {log}"],
    }


def _anchor_record_id(state: AgentStateV1, deps: list[str]) -> str | None:
    """Extracts the anchoring record_id from a completed vector task.

    The top chunk of the company-scoped vector search identifies the filing
    the question is actually about; graph retrieval is scoped to it so the
    math consumes values from the right document, not arbitrary filings.

    Year-aware preference: companies file maturity-schedule tables in MANY
    years (a 2003 filing also discusses "long-term debt maturities" — for
    2004-2008). When the question names fiscal years, prefer the first chunk
    whose text mentions one of them; fall back to the plain top chunk.
    """
    import re

    year_tokens = set(re.findall(r"\b(?:19|20)\d{2}\b", state.input))

    for dep in deps:
        env = state.sub_task_results.get(dep)
        if not isinstance(env, dict) or env.get("tool_name") != "vector_retrieval":
            continue
        chunks = (env.get("data") or {}).get("chunks") or []
        if not chunks:
            continue
        dated = [
            ch
            for ch in chunks
            if any(y in str(ch.get("text_content", "")) for y in year_tokens)
        ]
        candidates = [ch for ch in dated or chunks if ch.get("record_id")]
        if candidates:
            top_score = max(ch.get("similarity_score", 0) for ch in candidates)
            ids = {
                str(ch["record_id"])
                for ch in candidates
                if ch.get("similarity_score", 0) == top_score
            }
            if len(ids) == 1:
                return ids.pop()
    return None


def fan_out_router(state: AgentStateV1) -> list[Send] | str:
    """Evaluates pending sub-tasks and dynamically fans out independent branches via Send."""
    ready_tasks: list[dict[str, Any]] = []

    for call in state.tool_calls:
        if call.get("status") == "PENDING":
            deps = call.get("dependencies", [])

            if not deps or all(d in state.sub_task_results for d in deps):

                if call.get("target_tool") in {
                    "graph_retrieval",
                    "vector_retrieval",
                    "table_extract",
                }:
                    ready_tasks.append(call)

    if ready_tasks:
        sends: list[Send] = []
        for t in ready_tasks:
            payload = dict(t.get("payload", {}))
            if t.get("target_tool") == "graph_retrieval" and not payload.get(
                "record_id"
            ):

                anchor = state.record_id or _anchor_record_id(
                    state, t.get("dependencies", [])
                )
                if anchor:
                    payload["record_id"] = anchor
            sends.append(
                Send(
                    "sub_task_worker",
                    {
                        "task_id": t.get("task_id"),
                        "target_tool": t.get("target_tool"),
                        "payload": payload,
                    },
                )
            )
        return sends

    return "fuse_context"


def fan_out_router_node(state: AgentStateV1) -> dict[str, Any]:
    """No-op node for EDIT re-entry: exists so conditional edges from eval_risk
    can land on a node that re-dispatches the Send fan-out. Marks the risk
    re-check as complete for this pass."""
    return {
        "scratchpad": ["[Fan-Out] Re-dispatching edited sub-tasks under governance."]
    }


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
