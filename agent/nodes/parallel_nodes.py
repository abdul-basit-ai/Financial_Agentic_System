"""Parallel execution nodes for dynamic fan-out and fan-in using LangGraph Send API."""

from __future__ import annotations

from typing import Any

try:
    from langgraph.types import Send
except ImportError:  # pragma: no cover - version shims for very old langgraph
    try:
        from langgraph.constants import Send  # type: ignore[no-redef]
    except ImportError:  # type: ignore[no-redef]
        from langgraph.graph import Send  # type: ignore[attr-defined,no-redef]

from agent.nodes.concurrency import ToolSlot
from agent.state.schema import AgentStateV1
from agent.tools.graph_tool import GraphQueryInput, graph_retrieval_tool
from agent.tools.safe_math import SafeMathInput, safe_math_tool
from agent.tools.table_tool import TableExtractInput, table_extract_tool
from agent.tools.vector_tool import VectorSearchInput, vector_retrieval_tool


def sub_task_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Isolated worker node executing a single SubTask dispatched via Send API.

    Concurrency is bounded per-tool via the ToolSlot semaphore registry
    (agent.nodes.concurrency) — parallel Send branches queue rather than
    overwhelming downstream stores; wait/contention is logged for Phase 12.
    """
    task_id = payload.get("task_id", "task_unknown")
    target_tool = payload.get("target_tool")
    tool_payload = payload.get("payload", {})

    result_data: Any = None
    query_input: Any = None
    res: Any = None
    success = False
    error_msg: str | None = None
    log_entry = ""

    if target_tool not in {"graph_retrieval", "vector_retrieval", "safe_math", "table_extract"}:
        error_msg = f"Unsupported tool '{target_tool}' in sub_task_worker"
        log_entry = f"[Parallel Worker: {task_id}] Failed: {error_msg}"
        return {
            "sub_task_results": {
                task_id: {
                    "task_id": task_id,
                    "tool_name": target_tool,
                    "success": False,
                    "data": None,
                    "error": error_msg,
                }
            },
            "tool_results": [
                {
                    "task_id": task_id,
                    "tool_name": target_tool,
                    "success": False,
                    "data": None,
                    "error": error_msg,
                }
            ],
            "scratchpad": [log_entry],
        }

    with ToolSlot(target_tool) as slot:
        if slot.status == "timeout":
            # Hard backpressure: fail the task rather than execute unbounded.
            error_msg = (
                f"Concurrency limit timeout for '{target_tool}' after "
                f"{slot.wait_seconds:.1f}s of queuing"
            )
            log_entry = f"[Parallel Worker: {task_id}] Rejected: {error_msg}"
            envelope = {
                "task_id": task_id,
                "tool_name": target_tool,
                "success": False,
                "data": None,
                "error": error_msg,
            }
            return {
                "sub_task_results": {task_id: envelope},
                "tool_results": [envelope],
                "scratchpad": [log_entry],
            }

        contention_note = ""
        if slot.status == "waited":
            contention_note = f" (queued {slot.wait_seconds:.2f}s for slot)"

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
            log_entry = f"[Parallel Worker: {task_id}] Graph retrieved {count} records.{contention_note}"

        elif target_tool == "vector_retrieval":
            # Record anchoring, stage 1: restrict semantic search to the
            # anchored filing when known (planner injected it), else to the
            # company's own filings so the top chunk identifies the filing
            # the question is actually about (fan_out_router forwards that
            # record_id into the dependent graph_retrieval payloads).
            company = tool_payload.get("company_identifier")
            record_id = tool_payload.get("record_id")
            record_ids = None
            if not record_id and company:
                from agent.tools.graph_tool import list_company_record_ids

                record_ids = list_company_record_ids(company) or None
            query_input = VectorSearchInput(
                query_text=tool_payload.get("query_text", ""),
                top_k=tool_payload.get("top_k", 5),
                record_id=record_id,
                record_ids=record_ids,
            )
            res = vector_retrieval_tool(query_input)
            success = res.success
            result_data = res.data.model_dump() if res.data else None
            error_msg = res.error
            count = res.data.total_found if res.data else 0
            scope_note = " (company-scoped)" if record_ids else ""
            log_entry = f"[Parallel Worker: {task_id}] Vector retrieved {count} chunks{scope_note}.{contention_note}"

        elif target_tool == "table_extract":
            query_input = TableExtractInput(
                record_id=tool_payload.get("record_id", "")
            )
            res = table_extract_tool(query_input)
            success = res.success
            result_data = res.data.model_dump() if res.data else None
            error_msg = res.error
            count = res.data.total_values if res.data else 0
            log_entry = (
                f"[Parallel Worker: {task_id}] Table extracted "
                f"{count} values.{contention_note}"
            )

        elif target_tool == "safe_math":
            raw_expr = tool_payload.get("expression", "0")
            res = safe_math_tool(SafeMathInput(expression=raw_expr))
            success = res.success
            result_data = res.data.model_dump() if res.data else None
            error_msg = res.error
            val = res.data.formatted if res.data else "ERROR"
            log_entry = f"[Parallel Worker: {task_id}] Math calculated '{raw_expr}' -> {val}{contention_note}"

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
        if year_tokens:
            for ch in chunks:
                text = str(ch.get("text_content", ""))
                if any(y in text for y in year_tokens) and ch.get("record_id"):
                    return str(ch["record_id"])
        if chunks[0].get("record_id"):
            return str(chunks[0]["record_id"])
    return None


def fan_out_router(state: AgentStateV1) -> list[Send] | str:
    """Evaluates pending sub-tasks and dynamically fans out independent branches via Send."""
    ready_tasks: list[dict[str, Any]] = []

    for call in state.tool_calls:
        if call.get("status") == "PENDING":
            deps = call.get("dependencies", [])
            # Ready if all dependencies exist in sub_task_results
            if not deps or all(d in state.sub_task_results for d in deps):
                # Only fan out retrieval tasks (math is routed to compute)
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
                # Explicit caller anchor wins; vector-top-1 heuristic is the
                # fallback for free-form queries with no known filing.
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
