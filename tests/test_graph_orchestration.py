"""Integration test suite for Phase 6 LangGraph Orchestration.

Verifies:
1. End-to-end execution of single-hop financial retrieval queries.
2. End-to-end execution of multi-hop YoY calculation queries (retrieval -> compute -> synthesis).
3. Bounded cyclicity preventing infinite reasoning loops.
4. Checkpointer persistence across graph execution threads.
"""

from __future__ import annotations

import uuid
import pytest
from langgraph.checkpoint.memory import MemorySaver

from agent.graph import create_financial_agent
from agent.state.schema import AgentStateV1
from agent.tools.decomposer import decompose_query


@pytest.fixture
def agent_runner():
    saver = MemorySaver()
    return create_financial_agent(checkpointer=saver)


def test_single_hop_graph_orchestration(agent_runner) -> None:
    session_id = f"test_{uuid.uuid4().hex[:8]}"
    initial_state = AgentStateV1(
        input="What was Microsoft's total revenue in 2021?",
        company_identifier="MSFT",
    )

    config = {"configurable": {"thread_id": session_id}}
    final_output = agent_runner.invoke(initial_state.model_dump(), config=config)

    assert final_output["is_terminal"] is True
    assert final_output["final_answer"] is not None
    assert final_output["iteration_count"] >= 1
    assert len(final_output["scratchpad"]) > 0

    # Ensure memory promotion occurred
    assert any("Memory Promotion" in log for log in final_output["scratchpad"])


def test_multi_hop_yoy_calculation_orchestration(agent_runner) -> None:
    session_id = f"test_{uuid.uuid4().hex[:8]}"
    initial_state = AgentStateV1(
        input="What was the change in Apple net sales from 2002 to 2003?",
        company_identifier="AAPL",
    )

    config = {"configurable": {"thread_id": session_id}}
    final_output = agent_runner.invoke(initial_state.model_dump(), config=config)

    assert final_output["is_terminal"] is True
    assert final_output["final_answer"] is not None
    # Verify math calculation took place
    math_results = [r for r in final_output["tool_results"] if r.get("tool_name") == "safe_math"]
    assert len(math_results) >= 1
    assert math_results[0]["success"] is True


def test_graph_checkpointer_persistence(agent_runner) -> None:
    """Verifies that intermediate state snapshots are preserved by thread_id."""
    thread_id = f"thread_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = AgentStateV1(
        input="What was operating income for Apple in 2020?",
        company_identifier="AAPL",
    )

    # First run
    output_1 = agent_runner.invoke(initial_state.model_dump(), config=config)
    assert output_1["is_terminal"] is True

    # Retrieve checkpointed state directly from runner checkpointer
    checkpoint_state = agent_runner.get_state(config)
    assert checkpoint_state is not None
    assert checkpoint_state.values["input"] == initial_state.input
    assert checkpoint_state.values["trace_id"] == output_1["trace_id"]

# =====================================================================
# Zero-Evidence Synthesis Honesty Tests
# =====================================================================


def test_synthesizer_terminates_honestly_without_evidence(agent_runner) -> None:
    """Query targeting data absent from the graph must terminate at the
    iteration limit with an explicit insufficient-evidence answer - never a
    fabricated or falsely-successful response."""
    session_id = f"test_{uuid.uuid4().hex[:8]}"
    initial_state = AgentStateV1(
        input="What was the change in Amazon revenue from 2019 to 2020?",
        company_identifier="AMZN",  # sample data has no AMZN 2019/2020 values
    )
    config = {"configurable": {"thread_id": session_id}}
    final_output = agent_runner.invoke(initial_state.model_dump(), config=config)

    assert final_output["is_terminal"] is True
    assert final_output["final_answer"] is not None
    assert "Insufficient evidence" in final_output["final_answer"]
    assert final_output["iteration_count"] <= 4


def test_decomposer_routes_unanchored_metric_to_vector() -> None:
    """Metric-less queries (text-only values) must fan out to vector retrieval."""
    out = decompose_query("What portion of the estimate was used for equipment?")
    tools = {t.target_tool for t in out.sub_tasks}
    assert "vector_retrieval" in tools
    assert "graph_retrieval" in tools


def test_decomposer_yoy_without_metric_adds_narrative_backup() -> None:
    """YoY queries without a tabular metric hint schedule narrative backup."""
    out = decompose_query(
        "What was the change in the disclosed value from 2002 to 2003?",
        company="AAPL",
    )
    tools = [t.target_tool for t in out.sub_tasks]
    assert tools.count("graph_retrieval") == 2
    assert "vector_retrieval" in tools


def test_decomposer_clean_single_hop_stays_single() -> None:
    """Strong tabular-anchor query must NOT trigger extra vector fan-out."""
    out = decompose_query("What was Apple net sales in 2002?", company="AAPL")
    assert len(out.sub_tasks) == 1
    assert out.sub_tasks[0].target_tool == "graph_retrieval"
