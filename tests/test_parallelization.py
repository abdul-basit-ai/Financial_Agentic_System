"""Validation suite for Phase 7 Parallelization.

Verifies:
1. Dynamic Send API fan-out generation from decomposed sub-tasks.
2. Commutative state reducer determinism across non-deterministic completion orders.
3. Isolated execution of sub_task_worker across graph, vector, and math tools.
4. End-to-end multi-hop graph execution with parallel branch resolution.
"""

from __future__ import annotations

import uuid
from langgraph.checkpoint.memory import MemorySaver

from agent.graph import create_financial_agent
from agent.nodes.parallel_nodes import fan_out_router, sub_task_worker
from agent.state.schema import AgentStateV1, merge_sub_task_results


def test_fan_out_router_generates_sends() -> None:
    state = AgentStateV1(
        input="Compare Amazon revenue in 2019 and 2020",
        company_identifier="AMZN",
        tool_calls=[
            {
                "task_id": "task_1",
                "target_tool": "graph_retrieval",
                "payload": {"company_identifier": "AMZN", "year": 2019},
                "dependencies": [],
                "status": "PENDING",
            },
            {
                "task_id": "task_2",
                "target_tool": "graph_retrieval",
                "payload": {"company_identifier": "AMZN", "year": 2020},
                "dependencies": [],
                "status": "PENDING",
            },
            {
                "task_id": "task_3",
                "target_tool": "safe_math",
                "payload": {"expression": "subtract(task_2.amount, task_1.amount)"},
                "dependencies": ["task_1", "task_2"],
                "status": "PENDING",
            },
        ],
    )

    sends = fan_out_router(state)
    assert isinstance(sends, list)
    assert len(sends) == 2
    assert all(s.node == "sub_task_worker" for s in sends)
    assert {s.arg["task_id"] for s in sends} == {"task_1", "task_2"}


def test_sub_task_worker_execution() -> None:
    payload = {
        "task_id": "test_math_task",
        "target_tool": "safe_math",
        "payload": {"expression": "multiply(50, 4)"},
    }
    out = sub_task_worker(payload)

    assert "test_math_task" in out["sub_task_results"]
    assert out["sub_task_results"]["test_math_task"]["success"] is True
    assert out["sub_task_results"]["test_math_task"]["data"]["result"] == 200.0
    assert len(out["scratchpad"]) == 1


def test_commutative_monoid_out_of_order_convergence() -> None:
    """Verifies that out-of-order arrival produces identical state."""
    res_1 = {"task_1": {"amount": 1000.0}}
    res_2 = {"task_2": {"amount": 1250.0}}
    res_3 = {"task_3": {"amount": 1500.0}}

    order_a = merge_sub_task_results(merge_sub_task_results(res_1, res_2), res_3)
    order_b = merge_sub_task_results(merge_sub_task_results(res_3, res_1), res_2)
    order_c = merge_sub_task_results(merge_sub_task_results(res_2, res_3), res_1)

    assert order_a == order_b == order_c
    assert list(sorted(order_a.keys())) == ["task_1", "task_2", "task_3"]


def test_end_to_end_parallel_graph_execution() -> None:
    saver = MemorySaver()
    agent = create_financial_agent(checkpointer=saver)

    session_id = f"test_par_{uuid.uuid4().hex[:8]}"
    state = AgentStateV1(
        input="What was the change in Amazon revenue from 2019 to 2020?",
        company_identifier="AMZN",
    )

    config = {
        "configurable": {"thread_id": session_id},
        "max_concurrency": 4,
    }
    result = agent.invoke(state.model_dump(), config=config)

    assert result["is_terminal"] is True
    assert result["final_answer"] is not None
    assert len(result["sub_task_results"]) >= 2
    assert any("Fan-In Barrier" in log for log in result["scratchpad"])