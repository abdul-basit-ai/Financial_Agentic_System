"""Validation suite for Phase 5 State Schema, Reducers, and Memory Subsystems."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest

from agent.memory.episodic import compute_recency_score
from agent.memory.procedural import ProceduralMemoryBank
from agent.memory.promoter import MemoryPromotionEngine, calculate_episode_importance
from agent.memory.working import get_checkpointer
from agent.state.schema import (
    AgentStateV1,
    append_scratchpad,
    append_tool_results,
    merge_sub_task_results,
)
from agent.tools.memory_tool import MemoryQueryInput


def test_agent_state_v1_validation() -> None:
    state = AgentStateV1(
        input="What was Amazon's 2020 revenue?", company_identifier="AMZN"
    )
    assert state.trace_id is not None
    assert state.iteration_count == 0
    assert state.hitl_status == "NONE"

    with pytest.raises(ValueError):
        AgentStateV1(input="Invalid trace", trace_id="invalid-uuid")


def test_langgraph_reducers() -> None:

    s1 = ["Thought 1"]
    s2 = append_scratchpad(s1, "Thought 2")
    assert s2 == ["Thought 1", "Thought 2"]

    r1 = [{"tool": "math", "result": 100}]
    r2 = append_tool_results(r1, {"tool": "graph", "result": 200})
    assert len(r2) == 2

    branch_a = {"task_1": 1050.0}
    branch_b = {"task_2": 950.0}
    merged = merge_sub_task_results(branch_a, branch_b)
    assert merged == {"task_1": 1050.0, "task_2": 950.0}


def test_exponential_recency_decay() -> None:
    now = datetime.now(UTC)
    half_life = 30.0

    score_now = compute_recency_score(now, half_life_days=half_life)
    assert score_now == pytest.approx(1.0, rel=1e-3)

    past_30d = now - timedelta(days=30)
    score_30d = compute_recency_score(past_30d, half_life_days=half_life)
    assert score_30d == pytest.approx(0.5, rel=1e-3)

    past_60d = now - timedelta(days=60)
    score_60d = compute_recency_score(past_60d, half_life_days=half_life)
    assert score_60d == pytest.approx(0.25, rel=1e-3)


def test_procedural_archetype_matching() -> None:
    if os.getenv("FINAGENT_INTEGRATION_TESTS", "").strip() != "1":
        pytest.skip("Requires real SentenceTransformer model (offline test mode)")

    bank = ProceduralMemoryBank()

    arch, score = bank.match_archetype(
        "What was the percentage change in Microsoft revenue from 2020 to 2021?"
    )
    assert arch is not None
    assert arch.archetype_id == "yoy_delta_comparison"
    assert score > 0.55
    assert len(arch.planned_steps) == 3


def test_calculate_episode_importance() -> None:
    state = AgentStateV1(
        input="Complex financial calculation",
        tool_results=[{"tool": "graph"}, {"tool": "math"}],
        sub_task_results={"t1": 100, "t2": 200},
        hitl_status="APPROVED",
    )
    importance = calculate_episode_importance(state)
    assert 0.0 <= importance <= 1.0
    assert importance > 0.7


def test_memory_promotion_skips_incomplete_sessions() -> None:
    engine = MemoryPromotionEngine(episodic_store=None)
    empty_state = AgentStateV1(input="Unresolved prompt")
    promoted_id = engine.promote_session("sess_001", empty_state)
    assert promoted_id is None


def test_sqlite_checkpointer_persists_state_across_instances(
    tmp_path, monkeypatch
) -> None:
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict

    class S(TypedDict):
        value: str

    def node_a(state: S) -> dict:
        return {"value": "from_a"}

    monkeypatch.setenv("FINAGENT_CHECKPOINT_BACKEND", "sqlite")
    monkeypatch.setenv("FINAGENT_CHECKPOINT_PATH", str(tmp_path / "checkpoint.sqlite3"))
    cp = get_checkpointer()
    builder = StateGraph(S)
    builder.add_node("a", node_a)
    builder.add_edge(START, "a")
    builder.add_edge("a", END)
    graph = builder.compile(checkpointer=cp)

    cfg = {"configurable": {"thread_id": "test-thread-1"}}
    graph.invoke({"value": "init"}, cfg)
    from langgraph.checkpoint.sqlite import SqliteSaver

    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoint.sqlite3")) as reopened:
        snap = builder.compile(checkpointer=reopened).get_state(cfg)
    assert snap.values["value"] == "from_a"
    assert isinstance(cp, SqliteSaver)
    cp.conn.close()


def test_explicit_redis_backend_fails_closed_without_a_connection(monkeypatch) -> None:
    import redis

    monkeypatch.setenv("FINAGENT_CHECKPOINT_BACKEND", "redis")

    def unavailable(_client):
        raise redis.ConnectionError("Redis unavailable")

    monkeypatch.setattr(redis.Redis, "ping", unavailable)
    with pytest.raises(redis.ConnectionError):
        get_checkpointer(redis_url="redis://localhost:9999/0")


def test_memory_tool_schema_export() -> None:
    schema = MemoryQueryInput.model_json_schema()
    assert "properties" in schema
    assert "query_text" in schema["properties"]
