"""FastAPI routes for financial agent querying, streaming, and governance."""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from fastapi.responses import StreamingResponse

try:
    from langgraph.types import Command
except ImportError:
    Command = None

from agent.server.schemas import (
    ApprovalItem,
    ApprovalListResponse,
    AsyncJobResponse,
    HITLApprovalRequest,
    JobStatusResponse,
    QueryRequest,
    ThreadStateResponse,
)
from agent.server.streaming import stream_agent_events
from agent.state.schema import AgentStateV1

router = APIRouter(prefix="/api/v1", tags=["Financial Reasoning Agent"])

# In-memory repositories for tracking jobs and active thread identifiers
JOB_STORE: dict[str, dict[str, Any]] = {}
ACTIVE_THREADS: dict[str, dict[str, Any]] = {}


async def _execute_background_job(
    agent: Any,
    job_id: str,
    initial_state: AgentStateV1,
    config: dict[str, Any],
) -> None:
    """Asynchronous background execution worker."""
    JOB_STORE[job_id]["status"] = "RUNNING"
    JOB_STORE[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()

    try:
        final_state = await asyncio.to_thread(
            agent.invoke,
            initial_state.model_dump(),
            config=config,
        )
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        curr_state = agent.get_state(config)
        is_paused = bool(curr_state and curr_state.next)

        JOB_STORE[job_id].update({
            "status": "SUSPENDED_HITL" if is_paused else "COMPLETED",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "result": final_state,
            "execution_time_ms": round(duration_ms, 2),
        })
    except Exception as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        JOB_STORE[job_id].update({
            "status": "FAILED",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {str(exc)}",
            "execution_time_ms": round(duration_ms, 2),
        })


# =====================================================================
# 1. Real-Time Server-Sent Events (SSE) Endpoint
# =====================================================================


@router.post(
    "/query/stream",
    summary="Real-time Server-Sent Events (SSE) reasoning stream",
    response_class=StreamingResponse,
)
async def query_stream_endpoint(request: Request, body: QueryRequest) -> StreamingResponse:
    agent = request.app.state.agent
    initial_state = AgentStateV1(
        input=body.query,
        company_identifier=body.company_identifier,
    )
    config = {
        "configurable": {
            "thread_id": body.thread_id,
            "tenant_id": body.tenant_id,
        }
    }

    # Register active thread for state and approval inspection
    ACTIVE_THREADS[body.thread_id] = {
        "input_query": body.query,
        "company_identifier": body.company_identifier,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    return StreamingResponse(
        stream_agent_events(agent, initial_state, config),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Thread-Id": body.thread_id,
        },
    )


# =====================================================================
# 2. Asynchronous Job & Polling Endpoints
# =====================================================================


@router.post(
    "/query/async",
    response_model=AsyncJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit query for asynchronous decoupled execution",
)
async def query_async_endpoint(
    request: Request,
    body: QueryRequest,
    background_tasks: BackgroundTasks,
) -> AsyncJobResponse:
    agent = request.app.state.agent
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    now_str = datetime.now(timezone.utc).isoformat()

    initial_state = AgentStateV1(
        input=body.query,
        company_identifier=body.company_identifier,
    )
    config = {
        "configurable": {
            "thread_id": body.thread_id,
            "tenant_id": body.tenant_id,
        }
    }

    ACTIVE_THREADS[body.thread_id] = {
        "input_query": body.query,
        "company_identifier": body.company_identifier,
        "created_at": now_str,
    }

    JOB_STORE[job_id] = {
        "job_id": job_id,
        "thread_id": body.thread_id,
        "status": "QUEUED",
        "created_at": now_str,
        "updated_at": now_str,
        "result": None,
        "error": None,
        "execution_time_ms": None,
    }

    background_tasks.add_task(
        _execute_background_job,
        agent=agent,
        job_id=job_id,
        initial_state=initial_state,
        config=config,
    )

    return AsyncJobResponse(
        job_id=job_id,
        thread_id=body.thread_id,
        status="QUEUED",
        created_at=now_str,
        poll_url=f"/api/v1/jobs/{job_id}",
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll status of an asynchronous job",
)
async def get_job_status_endpoint(job_id: str) -> JobStatusResponse:
    if job_id not in JOB_STORE:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return JobStatusResponse(**JOB_STORE[job_id])


# =====================================================================
# 3. Governance: HITL Approvals Queue & Resumption Endpoints
# =====================================================================


@router.get(
    "/approvals",
    response_model=ApprovalListResponse,
    summary="List all execution threads currently paused awaiting HITL review",
)
async def list_approvals_endpoint(request: Request) -> ApprovalListResponse:
    agent = request.app.state.agent
    approval_items: list[ApprovalItem] = []

    for thread_id, meta in list(ACTIVE_THREADS.items()):
        config = {"configurable": {"thread_id": thread_id}}
        try:
            snapshot = agent.get_state(config)
        except Exception:
            continue

        if not snapshot or not snapshot.values:
            continue

        is_paused = bool(snapshot.next)
        hitl_status = snapshot.values.get("hitl_status", "NONE")

        # Select threads paused on interrupts or marked with PENDING HITL status
        if is_paused or hitl_status == "PENDING":
            scratchpad = snapshot.values.get("scratchpad", [])
            trigger_reasons = [line for line in scratchpad if line.strip().startswith("• [")]

            approval_items.append(
                ApprovalItem(
                    thread_id=thread_id,
                    trace_id=snapshot.values.get("trace_id", "unknown"),
                    input_query=meta.get("input_query", snapshot.values.get("input", "")),
                    company_identifier=meta.get("company_identifier"),
                    hitl_status=hitl_status,
                    paused_nodes=list(snapshot.next) if snapshot.next else [],
                    trigger_reasons=trigger_reasons,
                    pending_tool_calls=snapshot.values.get("tool_calls", []),
                    scratchpad_summary=scratchpad[-5:] if scratchpad else [],
                    created_at=meta.get("created_at", datetime.now(timezone.utc).isoformat()),
                )
            )

    return ApprovalListResponse(
        total_pending=len(approval_items),
        items=approval_items,
    )


@router.get(
    "/threads/{thread_id}/state",
    response_model=ThreadStateResponse,
    summary="Inspect state and check for active HITL suspensions",
)
async def get_thread_state_endpoint(request: Request, thread_id: str) -> ThreadStateResponse:
    agent = request.app.state.agent
    config = {"configurable": {"thread_id": thread_id}}

    snapshot = agent.get_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread '{thread_id}' has no recorded state.",
        )

    is_paused = bool(snapshot.next)
    interrupt_payload = None

    if hasattr(snapshot, "tasks") and snapshot.tasks:
        for task in snapshot.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                interrupt_payload = [i.value for i in task.interrupts]

    return ThreadStateResponse(
        thread_id=thread_id,
        is_paused=is_paused,
        next_nodes=list(snapshot.next),
        hitl_status=snapshot.values.get("hitl_status", "NONE"),
        state_values=snapshot.values,
        interrupt_payload=interrupt_payload,
    )


@router.post(
    "/threads/{thread_id}/resume",
    summary="Resume an interrupted thread with analyst approval or overrides",
)
async def resume_thread_endpoint(
    request: Request,
    thread_id: str,
    body: HITLApprovalRequest,
) -> dict[str, Any]:
    agent = request.app.state.agent
    config = {"configurable": {"thread_id": thread_id}}

    snapshot = agent.get_state(config)
    if not snapshot or not snapshot.next:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Thread '{thread_id}' is not in a paused or interruptible state.",
        )

    decision_payload = {
        "action": body.action,
        "analyst_id": body.analyst_id,
        "feedback": body.feedback,
        "overrides": body.overrides,
    }

    # Map the UI-facing alias "OVERRIDE" to the graph's "EDIT" verb. Without
    # this, an OVERRIDE submission fails closed to REJECT in hitl_gate.
    if decision_payload["action"] == "OVERRIDE":
        decision_payload["action"] = "EDIT"
        # UI OVERRIDE without tool_calls structure would be dropped by the
        # gate; pass overrides through untouched so the analyst's edits land.

    try:
        if Command is not None:
            resume_cmd = Command(resume=decision_payload)
            resumed_output = await asyncio.to_thread(agent.invoke, resume_cmd, config=config)
        else:
            agent.update_state(config, {"hitl_status": body.action})
            resumed_output = await asyncio.to_thread(agent.invoke, None, config=config)

        return {
            "thread_id": thread_id,
            "status": "RESUMED",
            "action_applied": body.action,
            "final_answer": resumed_output.get("final_answer"),
            "is_terminal": resumed_output.get("is_terminal", False),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume thread: {str(exc)}",
        )


@router.get("/health", summary="Health check endpoint")
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "finagent-production-gateway"}