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

    # Core Execution Context
    trace_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Immutable distributed trace identifier (UUIDv4)",
    )
    input: str = Field(..., description="Raw user query or financial task prompt")
    company_identifier: str | None = Field(
        default=None, description="Primary company target extracted from context"
    )
    iteration_count: int = Field(
        default=0, ge=0, description="Number of completed graph reasoning cycles"
    )

    # Working Memory & Reasoning Buffers (LangGraph Reducer-Annotated)
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

    # Memory & Context References
    memory_refs: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Episodic, semantic, or procedural facts loaded during initialization",
    )

    # Human-in-the-Loop & Execution Control
    hitl_status: Literal["NONE", "PENDING", "APPROVED", "APPROVE", "REJECTED", "REJECT", "EDIT"] = Field(
        default="NONE", description="Phase 8 approval gate status"
    )
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
            uuid.UUID(v, version=4)
        except ValueError as exc:
            raise ValueError(f"trace_id must be a valid UUIDv4 string: {v}") from exc
        return v


# =====================================================================
# Schema Versioning — V1 → V2 Migration Note
# =====================================================================
# AgentStateV1 is the frozen contract for Phases 6-14 (all graph nodes read/
# write it; the Redis checkpointer serializes model_dump_json of it).
#
# V2 EXTENSION RULES (how V2 would evolve without breaking checkpoints):
# 1. ADDITIVE ONLY within a release: new fields must be optional with defaults
#    (e.g. `cost_budget_usd: float = Field(default=0.0)`) so V1-serialized
#    checkpoint payloads validate under V2 (pydantic ignores absent keys).
# 2. NEVER rename or retype existing fields; deprecate by adding a parallel
#    field and a validator shim (old -> new) for one release.
# 3. Reducers are part of the contract: a field's Annotated reducer must keep
#    its associativity/identity semantics or parallel fan-in (Phase 7) breaks.
# 4. Changing a field's shape requires a checkpoint re-write step: bump
#    GRAPH_VERSION in agent/nodes/plan_node.py, write a migration in
#    agent/state/migrations.py that loads old JSON -> transforms -> saves,
#    and invalidate (do not silently drop) stale threads.
# 5. hitl_status / risk_evaluated are governance state: any V2 change must be
#    re-reviewed by the Phase 8 pause/resume tests and the audit ledger.
# =====================================================================