"""Agent state schema definitions and LangGraph reducers."""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def append_scratchpad(left: list[str], right: list[str] | str) -> list[str]:
    """Append-only reducer for agent chain-of-thought scratchpad entries."""
    if isinstance(right, str):
        return left + [right]
    return left + list(right)


def append_tool_results(
    left: list[dict[str, Any]], right: list[dict[str, Any]] | dict[str, Any]
) -> list[dict[str, Any]]:
    """Append-only reducer for completed tool outputs."""
    if isinstance(right, dict):
        return left + [right]
    return left + list(right)


def append_llm_usage(
    left: list[dict[str, Any]], right: list[dict[str, Any]] | dict[str, Any]
) -> list[dict[str, Any]]:
    """Append-only reducer for provider token/cost telemetry."""
    if isinstance(right, dict):
        return left + [right]
    return left + list(right)


def merge_sub_task_results(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    """Monotonic dictionary merge reducer for Phase 7 parallel branch fan-in."""
    merged = dict(left)
    merged.update(right)
    return merged


class AgentStateV1(BaseModel):
    """Strict typed contract for LangGraph node read/write operations."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    trace_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Immutable distributed trace identifier (UUIDv4)",
    )
    input: str = Field(..., description="Raw user query or financial task prompt")
    owner_id: str = "local"
    document_ids: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    fused_context: str = ""
    answer_status: Literal["grounded", "insufficient", "rejected"] | None = None
    answer_task_id: str | None = None
    company_identifier: str | None = Field(
        default=None, description="Primary company target extracted from context"
    )
    record_id: str | None = Field(
        default=None,
        description="Anchoring FinQA record id (the filing the question is about), "
        "when the caller knows it — scopes retrieval to one document",
    )
    iteration_count: int = Field(
        default=0, ge=0, description="Number of completed graph reasoning cycles"
    )

    scratchpad: Annotated[list[str], append_scratchpad] = Field(
        default_factory=list,
        description="Internal chain-of-thought and validation log buffer",
    )
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list, description="Queue of pending tool dispatch operations"
    )
    tool_results: Annotated[list[dict[str, Any]], append_tool_results] = Field(
        default_factory=list, description="Accumulated tool execution result envelopes"
    )
    sub_task_results: Annotated[dict[str, Any], merge_sub_task_results] = Field(
        default_factory=dict,
        description="Thread-safe accumulator for Phase 7 parallel fan-in",
    )
    llm_usage: Annotated[list[dict[str, Any]], append_llm_usage] = Field(
        default_factory=list,
        description="Per-call provider token and estimated cost telemetry",
    )

    memory_refs: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Episodic, semantic, or procedural facts loaded during initialization",
    )

    hitl_status: Literal[
        "NONE", "PENDING", "APPROVED", "APPROVE", "REJECTED", "REJECT", "EDIT"
    ] = Field(default="NONE", description="Phase 8 approval gate status")
    risk_evaluated: bool = Field(
        default=False,
        description="True once eval_risk ran in the current iteration (prevents re-eval loops after compute)",
    )
    final_answer: str | None = Field(
        default=None, description="Synthesized final financial answer"
    )
    is_terminal: bool = Field(
        default=False, description="Flag indicating graph termination"
    )

    @field_validator("trace_id")
    @classmethod
    def validate_trace_id(cls, v: str) -> str:
        try:
            if uuid.UUID(v).version != 4:
                raise ValueError("wrong UUID version")
        except ValueError as exc:
            raise ValueError(f"trace_id must be a valid UUIDv4 string: {v}") from exc
        return v
