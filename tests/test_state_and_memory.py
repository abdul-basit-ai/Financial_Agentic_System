"""Validation suite for Phase 5 State Schema, Reducers, and Memory Subsystems."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
import pytest

from agent.memory.episodic import (
    EpisodicEntry,
    EpisodicMemoryStore,
    compute_recency_score,
)
from agent.memory.procedural import ProceduralMemoryBank
from agent.memory.promoter import (
    MemoryPromotionEngine,
    calculate_episode_importance,
)
from agent.memory.working import WorkingMemoryManager
from agent.state.schema import (
    AgentStateV1,
    append_scratchpad,
    append_tool_results,
    merge_sub_task_results,
)
from agent.tools.memory_tool import MemoryQueryInput, MemoryRetrievalTool


# =====================================================================
# State Schema & Reducer Tests
# =====================================================================


def test_agent_state_v1_validation() -> None:
    state = AgentStateV1(input="What was Amazon's 2020 revenue?", company_identifier="AMZN")
    assert state.trace_id is not None
    assert state.iteration_count == 0
    assert state.hitl_status == "NONE"

    # Invariant: Invalid UUIDv4 must fail validation
    with pytest.raises(ValueError):
        AgentStateV1(input="Invalid trace", trace_id="invalid-uuid")


def test_langgraph_reducers() -> None:
    # 1. Scratchpad Reducer
    s1 = ["Thought 1"]
    s2 = append_scratchpad(s1, "Thought 2")
    assert s2 == ["Thought 1", "Thought 2"]

    # 2. Tool Results Reducer
    r1 = [{"tool": "math", "result": 100}]
    r2 = append_tool_results(r1, {"tool": "graph", "result": 200})
    assert len(r2) == 2

    # 3. Parallel Branch Fan-In Merge Reducer (Phase 7 Monotonicity)
    branch_a = {"task_1": 1050.0}
    branch_b = {"task_2": 950.0}
    merged = merge_sub_task_results(branch_a, branch_b)
    assert merged == {"task_1": 1050.0, "task_2": 950.0}


# =====================================================================
# Episodic Decay Mathematics Tests
# =====================================================================


def test_exponential_recency_decay() -> None:
    now = datetime.now(timezone.utc)
    half_life = 30.0

    # Event occurring now: Score = 2^0 = 1.0
    score_now = compute_recency_score(now, half_life_days=half_life)
    assert score_now == pytest.approx(1.0, rel=1e-3)

    # Event occurring exactly 30 days ago: Score = 2^(-1) = 0.5
    past_30d = now - timedelta(days=30)
    score_30d = compute_recency_score(past_30d, half_life_days=half_life)
    assert score_30d == pytest.approx(0.5, rel=1e-3)

    # Event occurring 60 days ago: Score = 2^(-2) = 0.25
    past_60d = now - timedelta(days=60)
    score_60d = compute_recency_score(past_60d, half_life_days=half_life)
    assert score_60d == pytest.approx(0.25, rel=1e-3)


# =====================================================================
# Procedural Memory Archetype Matching Tests
# =====================================================================


def test_procedural_archetype_matching() -> None:
    bank = ProceduralMemoryBank()

    # Query matching YoY comparison archetype
    arch, score = bank.match_archetype(
        "What was the percentage change in Microsoft revenue from 2020 to 2021?"
    )
    assert arch is not None
    assert arch.archetype_id == "yoy_delta_comparison"
    assert score > 0.55
    assert len(arch.planned_steps) == 3


# =====================================================================
# Memory Promotion & Importance Tests
# =====================================================================


def test_calculate_episode_importance() -> None:
    state = AgentStateV1(
        input="Complex financial calculation",
        tool_results=[{"tool": "graph"}, {"tool": "math"}],
        sub_task_results={"t1": 100, "t2": 200},
        hitl_status="APPROVED",
    )
    importance = calculate_episode_importance(state)
    assert 0.0 <= importance <= 1.0
    assert importance > 0.7  # High complexity + HITL approval yields high importance


def test_memory_promotion_skips_incomplete_sessions() -> None:
    engine = MemoryPromotionEngine(episodic_store=None)
    empty_state = AgentStateV1(input="Unresolved prompt")
    promoted_id = engine.promote_session("sess_001", empty_state)
    assert promoted_id is None


# =====================================================================
# JSON Schema Export Test (Phase 10 MCP Readiness)
# =====================================================================


def test_memory_tool_schema_export() -> None:
    schema = MemoryQueryInput.model_json_schema()
    assert "properties" in schema
    assert "query_text" in schema["properties"]