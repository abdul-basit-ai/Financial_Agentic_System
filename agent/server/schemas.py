"""Pydantic schemas for HTTP endpoints, async jobs, SSE streaming, and HITL governance."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    """Payload for initiating a financial reasoning query."""

    model_config = ConfigDict(extra="ignore")

    query: str = Field(..., description="Natural language financial question or instruction")
    company_identifier: str | None = Field(
        default=None, description="Company ticker or identifier (e.g., 'AAPL', 'AMZN')"
    )
    thread_id: str = Field(
        default_factory=lambda: f"thread_{uuid.uuid4().hex[:12]}",
        description="Persistent conversation/execution thread ID for checkpointing",
    )
    tenant_id: str = Field(default="default", description="Tenant isolation identifier")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary caller-provided metadata tags"
    )


class AsyncJobResponse(BaseModel):
    """Immediate acknowledgment for long-running asynchronous execution."""

    job_id: str
    thread_id: str
    status: Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED", "SUSPENDED_HITL"]
    created_at: str
    poll_url: str


class JobStatusResponse(BaseModel):
    """Status check response for polled background jobs."""

    job_id: str
    thread_id: str
    status: Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED", "SUSPENDED_HITL"]
    created_at: str
    updated_at: str
    result: dict[str, Any] | None = None
    error: str | None = None
    execution_time_ms: float | None = None


class HITLApprovalRequest(BaseModel):
    """Analyst review decision payload to resume an interrupted state machine."""

    # NOTE: the graph's hitl_gate understands APPROVE / REJECT / EDIT (an EDIT
    # must carry overrides.tool_calls). "OVERRIDE" here is a UI-facing alias
    # that must map to EDIT, otherwise the gate fails closed to REJECT.
    action: Literal["APPROVE", "REJECT", "EDIT", "OVERRIDE"] = Field(
        ..., description="Compliance review action"
    )
    analyst_id: str = Field(default="analyst_system", description="Reviewer identifier")
    feedback: str = Field(default="", description="Review justification or critique notes")
    overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="State delta overrides (e.g. corrected line items, altered parameters)",
    )


class ThreadStateResponse(BaseModel):
    """Snapshot inspection response for active, paused, or finished threads."""

    thread_id: str
    is_paused: bool
    next_nodes: list[str]
    hitl_status: str
    state_values: dict[str, Any]
    interrupt_payload: Any | None = None


class ApprovalItem(BaseModel):
    """Structured record for a thread currently suspended awaiting HITL review."""

    thread_id: str
    trace_id: str = "unknown"
    input_query: str = ""
    company_identifier: str | None = None
    hitl_status: str = "PENDING"
    paused_nodes: list[str] = Field(default_factory=list)
    trigger_reasons: list[str] = Field(default_factory=list)
    pending_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    scratchpad_summary: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ApprovalListResponse(BaseModel):
    """Queue of active approval requests for the compliance interface."""

    total_pending: int
    items: list[ApprovalItem]


class SSEEvent(BaseModel):
    """Structured event envelope for Server-Sent Events."""

    event: str
    data: dict[str, Any]

    def to_sse_packet(self) -> str:
        import json

        json_data = json.dumps(self.data, default=str)
        return f"event: {self.event}\ndata: {json_data}\n\n"