from __future__ import annotations

import asyncio
import os
import time
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from agent.documents.service import DocumentService
from agent.server.auth import Principal, auth_mode, require_principal, supabase_request
from agent.server.schemas import (
    ApprovalItem,
    ApprovalListResponse,
    AsyncJobResponse,
    HITLApprovalRequest,
    JobStatusResponse,
    QueryRequest,
    SSEEvent,
    ThreadStateResponse,
)
from agent.server.store import AdmissionLimitError, RecordStore, Transaction

router = APIRouter(prefix="/api/v1")
Identity = Annotated[Principal, Depends(require_principal)]


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=1024)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=1, max_length=8192)


def store_for(request: Request) -> RecordStore:
    return cast(RecordStore, request.app.state.store)


def documents_for(request: Request) -> DocumentService:
    return cast(DocumentService, request.app.state.documents)


def _admit(tx: Transaction) -> None:
    try:
        tx.admit()
    except AdmissionLimitError as exc:
        raise HTTPException(429, str(exc), headers={"Retry-After": "60"}) from exc


@router.get("/config")
def public_config() -> dict[str, Any]:
    return {
        "auth_mode": auth_mode(),
        "supabase_url": os.getenv("SUPABASE_URL", ""),
        "supabase_publishable_key": os.getenv("SUPABASE_PUBLISHABLE_KEY", ""),
        "max_upload_bytes": int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024))),
        "features": {"documents": True},
    }


def _exchange(grant: str, payload: dict[str, str]) -> dict[str, Any]:
    if auth_mode() != "supabase":
        raise HTTPException(400, "Use your worker access token.")
    response = supabase_request(
        "POST", f"/auth/v1/token?grant_type={grant}", json=payload
    )
    if response.status_code != 200:
        raise HTTPException(
            401 if response.status_code in {400, 401, 403, 422} else 503,
            "Sign-in failed. Check your credentials and try again.",
        )
    try:
        data = response.json()
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("access_token"), str)
            or not data["access_token"]
        ):
            raise ValueError("Invalid token response")
    except ValueError as exc:
        raise HTTPException(503, "Authentication is temporarily unavailable.") from exc
    return {
        key: data[key]
        for key in ("access_token", "refresh_token", "expires_in", "user")
        if key in data
    }


@router.post("/auth/login")
def login(body: LoginRequest) -> dict[str, Any]:
    return _exchange("password", body.model_dump())


@router.post("/auth/refresh")
def refresh(body: RefreshRequest) -> dict[str, Any]:
    return _exchange("refresh_token", body.model_dump())


def _submit(
    request: Request, body: QueryRequest, identity: Principal
) -> dict[str, Any]:
    store = store_for(request)
    now = datetime.now(UTC).isoformat()
    thread: dict[str, Any] = {
        "thread_id": body.thread_id,
        "input_query": body.query,
        "company_identifier": body.company_identifier,
        "document_ids": body.document_ids,
        "created_at": now,
        "is_paused": False,
        "state_values": {},
        "next_nodes": [],
    }
    job: dict[str, Any] = {
        "job_id": f"job_{uuid.uuid4().hex}",
        "thread_id": body.thread_id,
        "kind": "query",
        "status": "QUEUED",
        "created_at": now,
        "updated_at": now,
        "query_payload": body.model_dump(),
        "result": None,
        "error": None,
    }
    with store.transaction(identity.user_id) as tx:
        if tx.get("threads", body.thread_id):
            raise HTTPException(409, "This analysis ID has already been used.")
        for document_id in body.document_ids:
            if (tx.get("documents", document_id) or {}).get("status") != "ready":
                raise HTTPException(
                    404, "Every selected report must be available and ready."
                )
        _admit(tx)
        tx.put("threads", body.thread_id, thread)
        tx.put("jobs", job["job_id"], job)
    return job


@router.post("/query/async", status_code=202, response_model=AsyncJobResponse)
def query_async_endpoint(
    request: Request, body: QueryRequest, identity: Identity
) -> AsyncJobResponse:
    job = _submit(request, body, identity)
    return AsyncJobResponse(
        job_id=job["job_id"],
        thread_id=body.thread_id,
        status="QUEUED",
        created_at=job["created_at"],
        poll_url=f"/api/v1/jobs/{job['job_id']}",
    )


@router.post("/query/stream")
async def query_stream_endpoint(
    request: Request, body: QueryRequest, identity: Identity
) -> StreamingResponse:
    job = await asyncio.to_thread(_submit, request, body, identity)

    async def events():
        yield SSEEvent(
            event="lifecycle",
            data={
                "stage": "INITIATED",
                "thread_id": body.thread_id,
                "job_id": job["job_id"],
            },
        ).to_sse_packet()
        yield SSEEvent(
            event="node_update",
            data={
                "node": "retrieve_evidence",
                "latest_log": "Searching selected sources and validating cited inputs.",
            },
        ).to_sse_packet()
        start = time.monotonic()
        deadline = float(os.getenv("QUERY_STREAM_TIMEOUT", "180"))
        while time.monotonic() - start < deadline:
            if await request.is_disconnected():
                return
            current = await asyncio.to_thread(
                store_for(request).get, "jobs", job["job_id"], identity.user_id
            )
            if current and current["status"] in {
                "COMPLETED",
                "SUSPENDED_HITL",
                "FAILED",
                "CANCELLED",
            }:
                if current["status"] == "FAILED":
                    yield SSEEvent(
                        event="error",
                        data={
                            "message": current.get("error", "Analysis failed."),
                            "job_id": job["job_id"],
                        },
                    ).to_sse_packet()
                elif current["status"] == "SUSPENDED_HITL":
                    thread = (
                        store_for(request).get(
                            "threads", body.thread_id, identity.user_id
                        )
                        or {}
                    )
                    yield SSEEvent(
                        event="interrupt",
                        data={
                            "thread_id": body.thread_id,
                            "paused_at_nodes": thread.get("next_nodes", []),
                        },
                    ).to_sse_packet()
                else:
                    result = current.get("result") or {}
                    yield SSEEvent(
                        event="final_answer",
                        data={
                            "thread_id": body.thread_id,
                            **{
                                k: result.get(k)
                                for k in (
                                    "final_answer",
                                    "citations",
                                    "answer_status",
                                    "calculation",
                                    "llm_usage",
                                )
                            },
                        },
                    ).to_sse_packet()
                    yield SSEEvent(
                        event="complete",
                        data={"status": "FINISHED", "thread_id": body.thread_id},
                    ).to_sse_packet()
                return
            yield ": keep-alive\n\n"
            await asyncio.sleep(0.25)
        yield SSEEvent(
            event="error",
            data={
                "message": "Analysis is still running. Use the saved analysis ID to recover the result.",
                "job_id": job["job_id"],
                "thread_id": body.thread_id,
            },
        ).to_sse_packet()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Thread-Id": body.thread_id,
        },
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status_endpoint(
    request: Request, job_id: str, identity: Identity
) -> JobStatusResponse:
    job = store_for(request).get("jobs", job_id, identity.user_id)
    if job is None:
        raise HTTPException(404, "Analysis not found.")
    return JobStatusResponse(**job)


@router.get("/approvals", response_model=ApprovalListResponse)
def list_approvals_endpoint(
    request: Request, identity: Identity
) -> ApprovalListResponse:
    items = []
    for thread in store_for(request).list(
        "threads", identity.user_id, limit=None, paused_only=True
    ):
        values = thread.get("state_values", {})
        scratchpad = values.get("scratchpad", [])
        items.append(
            ApprovalItem(
                thread_id=thread["thread_id"],
                trace_id=values.get("trace_id", "unknown"),
                input_query=thread.get("input_query", ""),
                company_identifier=thread.get("company_identifier"),
                hitl_status=values.get("hitl_status", "PENDING"),
                paused_nodes=thread.get("next_nodes", []),
                pending_tool_calls=values.get("tool_calls", []),
                trigger_reasons=[line for line in scratchpad if "[" in line][-5:],
                scratchpad_summary=scratchpad[-5:],
                created_at=thread["created_at"],
            )
        )
    return ApprovalListResponse(total_pending=len(items), items=items)


@router.get("/threads/{thread_id}/state", response_model=ThreadStateResponse)
def get_thread_state_endpoint(
    request: Request, thread_id: str, identity: Identity
) -> ThreadStateResponse:
    thread = store_for(request).get("threads", thread_id, identity.user_id)
    if thread is None:
        raise HTTPException(404, "Analysis not found.")
    values = thread.get("state_values", {})
    return ThreadStateResponse(
        thread_id=thread_id,
        is_paused=thread.get("is_paused", False),
        next_nodes=thread.get("next_nodes", []),
        hitl_status=values.get("hitl_status", "NONE"),
        state_values=values,
    )


@router.get("/threads")
def list_threads_endpoint(request: Request, identity: Identity) -> dict[str, Any]:
    records = store_for(request).list("threads", identity.user_id, limit=100)
    fields = (
        "thread_id",
        "input_query",
        "company_identifier",
        "document_ids",
        "created_at",
        "updated_at",
        "is_paused",
    )
    return {
        "items": [{key: record.get(key) for key in fields} for record in records],
        "total": len(records),
    }


def _queue_resume(
    store: RecordStore, thread_id: str, body: HITLApprovalRequest, identity: Principal
) -> str:
    with store.transaction(identity.user_id) as tx:
        thread = tx.get("threads", thread_id)
        if not thread:
            raise HTTPException(404, "Analysis not found.")
        if not thread.get("is_paused"):
            raise HTTPException(400, "This analysis is not awaiting review.")
        review = str(
            thread.get("review_id")
            or thread.get("state_values", {}).get("iteration_count", 0)
        )
        job_id = "resume_" + thread_id + "_" + review
        previous = tx.get("jobs", job_id)
        if previous and previous.get("status") != "FAILED":
            raise HTTPException(409, "A decision for this review is already queued.")
        _admit(tx)
        now = datetime.now(UTC).isoformat()
        decision = {
            **body.model_dump(),
            "action": "EDIT" if body.action == "OVERRIDE" else body.action,
            "analyst_id": identity.user_id,
        }
        job = {
            "job_id": job_id,
            "thread_id": thread_id,
            "kind": "resume",
            "status": "QUEUED",
            "created_at": now,
            "updated_at": now,
            "decision": decision,
            "checkpoint_id": thread.get("checkpoint_id"),
            "query_payload": {
                "query": thread.get("input_query", ""),
                "document_ids": [],
            },
        }
        tx.put("jobs", job_id, job)
        return job_id


@router.post("/threads/{thread_id}/resume")
async def resume_thread_endpoint(
    request: Request, thread_id: str, body: HITLApprovalRequest, identity: Identity
) -> dict[str, Any]:
    store = store_for(request)
    job_id = await asyncio.to_thread(_queue_resume, store, thread_id, body, identity)
    for _ in range(100):
        current = await asyncio.to_thread(store.get, "jobs", job_id, identity.user_id)
        if current and current["status"] in {"COMPLETED", "FAILED", "SUSPENDED_HITL"}:
            if current["status"] == "FAILED":
                raise HTTPException(500, current.get("error", "Resume failed."))
            result = current.get("result") or {}
            return {
                "thread_id": thread_id,
                "status": "RESUMED",
                "action_applied": body.action,
                **{
                    key: result.get(key)
                    for key in (
                        "final_answer",
                        "citations",
                        "answer_status",
                        "calculation",
                        "is_terminal",
                    )
                },
            }
        await asyncio.sleep(0.2)
    return {
        "thread_id": thread_id,
        "status": "QUEUED",
        "job_id": job_id,
        "is_terminal": False,
    }


@router.get("/documents")
def list_documents(request: Request, identity: Identity) -> dict[str, Any]:
    rows = store_for(request).list("documents", identity.user_id, limit=None)
    return {
        "items": [
            {k: v for k, v in row.items() if k not in {"storage_key", "deletion_token"}}
            for row in rows
        ],
        "total": len(rows),
    }


@router.post("/documents", status_code=202)
async def upload_document(
    request: Request,
    identity: Identity,
    file: UploadFile = File(...),
    company_identifier: str | None = Form(None),
) -> dict[str, Any]:
    cap = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
    content = await file.read(cap + 1)
    await file.close()
    if len(content) > cap:
        raise HTTPException(413, "The report exceeds the upload size limit.")
    try:
        record = await asyncio.to_thread(
            documents_for(request).create,
            identity.user_id,
            file.filename or "report.pdf",
            content,
            company_identifier,
        )
    except AdmissionLimitError as exc:
        raise HTTPException(429, str(exc), headers={"Retry-After": "60"}) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        k: v for k, v in record.items() if k not in {"storage_key", "deletion_token"}
    }


def _document(
    request: Request, document_id: str, identity: Principal
) -> dict[str, Any]:
    row = store_for(request).get("documents", document_id, identity.user_id)
    if row is None:
        raise HTTPException(404, "Report not found.")
    return row


@router.get("/documents/{document_id}")
def get_document(
    request: Request, document_id: str, identity: Identity
) -> dict[str, Any]:
    return {
        k: v
        for k, v in _document(request, document_id, identity).items()
        if k not in {"storage_key", "deletion_token"}
    }


@router.get("/documents/{document_id}/evidence")
def document_evidence(
    request: Request, document_id: str, identity: Identity
) -> dict[str, Any]:
    _document(request, document_id, identity)
    try:
        items = documents_for(request).evidence(identity.user_id, [document_id])
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"items": [item.citation() for item in items], "total": len(items)}


@router.get("/documents/{document_id}/download")
def download_document(
    request: Request, document_id: str, identity: Identity
) -> Response:
    record = _document(request, document_id, identity)
    if record.get("status") == "deleting":
        raise HTTPException(409, "Report deletion is in progress.")
    return Response(
        documents_for(request).storage.read(record["storage_key"]),
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'attachment; filename="report.pdf"',
            "Cache-Control": "private, no-store",
        },
    )


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(request: Request, document_id: str, identity: Identity) -> Response:
    record = _document(request, document_id, identity)
    if record.get("status") in {"queued", "processing"}:
        raise HTTPException(
            409, "Wait for report processing to finish before deleting it."
        )
    try:
        documents_for(request).delete(identity.user_id, document_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return Response(status_code=204)


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "financial-document-worker"}
