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


def test_fan_out_router_injects_anchor_record_id() -> None:
    """Record anchoring: a graph task depending on a completed vector task
    must receive the vector top chunk's record_id in its dispatch payload,
    so retrieval reads the filing the question is actually about."""
    state = AgentStateV1(
        input="What was the change in ETR long-term debt maturities from 2016 to 2017?",
        company_identifier="ETR",
        tool_calls=[
            {
                "task_id": "task_1",
                "target_tool": "vector_retrieval",
                "payload": {
                    "query_text": "long-term debt maturities",
                    "company_identifier": "ETR",
                },
                "dependencies": [],
                "status": "COMPLETED",
            },
            {
                "task_id": "task_2",
                "target_tool": "graph_retrieval",
                "payload": {"company_identifier": "ETR", "year": 2016},
                "dependencies": ["task_1"],
                "status": "PENDING",
            },
        ],
        sub_task_results={
            "task_1": {
                "task_id": "task_1",
                "tool_name": "vector_retrieval",
                "success": True,
                "data": {
                    "chunks": [
                        {
                            "record_id": "ETR/2017/page_422.pdf-2",
                            "text_content": "maturities...",
                            "similarity_score": 0.62,
                        }
                    ],
                    "total_found": 1,
                },
                "error": None,
            }
        },
    )

    sends = fan_out_router(state)
    assert isinstance(sends, list)
    assert len(sends) == 1
    assert sends[0].arg["payload"]["record_id"] == "ETR/2017/page_422.pdf-2"


def test_fan_out_router_no_anchor_no_injection() -> None:
    """Without a usable anchor (vector task empty), the graph payload must be
    dispatched unchanged — the retrieval ladder inside the tool remains the
    safety net."""
    state = AgentStateV1(
        input="Any question",
        company_identifier="ETR",
        tool_calls=[
            {
                "task_id": "task_1",
                "target_tool": "vector_retrieval",
                "payload": {"query_text": "x"},
                "dependencies": [],
                "status": "COMPLETED",
            },
            {
                "task_id": "task_2",
                "target_tool": "graph_retrieval",
                "payload": {"company_identifier": "ETR", "year": 2016},
                "dependencies": ["task_1"],
                "status": "PENDING",
            },
        ],
        sub_task_results={
            "task_1": {
                "task_id": "task_1",
                "tool_name": "vector_retrieval",
                "success": True,
                "data": {"chunks": [], "total_found": 0},
                "error": None,
            }
        },
    )

    sends = fan_out_router(state)
    assert isinstance(sends, list)
    assert "record_id" not in sends[0].arg["payload"]


def test_anchor_extraction_from_graph_dependency_fails_closed() -> None:
    """A graph dependency that is NOT a vector task must not be mined for an
    anchor (chunks only exist on vector results)."""
    state = AgentStateV1(
        input="q",
        company_identifier="ETR",
        tool_calls=[
            {
                "task_id": "task_2",
                "target_tool": "graph_retrieval",
                "payload": {"company_identifier": "ETR"},
                "dependencies": ["task_1"],
                "status": "PENDING",
            },
        ],
        sub_task_results={
            "task_1": {
                "task_id": "task_1",
                "tool_name": "graph_retrieval",
                "success": True,
                "data": {
                    "records": [{"row_label": "revenue", "amount": 5.0}],
                    "total_found": 1,
                },
                "error": None,
            }
        },
    )

    sends = fan_out_router(state)
    assert isinstance(sends, list)
    assert "record_id" not in sends[0].arg["payload"]


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
    }
    result = agent.invoke(state.model_dump(), config=config)

    assert result["is_terminal"] is True
    assert result["final_answer"] is not None
    assert len(result["sub_task_results"]) >= 2
    assert any("Fan-In Barrier" in log for log in result["scratchpad"])


# =====================================================================
# Per-Tool Concurrency Limiter Tests (Phase 7 hard backpressure)
# =====================================================================


def test_tool_slot_immediate_and_release() -> None:
    from agent.nodes.concurrency import ToolSlot

    with ToolSlot("safe_math") as slot:
        assert slot.status == "immediate"
        assert slot.wait_seconds == 0.0
    # Released -> next acquire is immediate again
    with ToolSlot("safe_math") as slot2:
        assert slot2.status == "immediate"


def test_tool_slot_blocks_at_limit_and_queues() -> None:
    import threading

    from agent.nodes.concurrency import ToolSlot, get_contention_stats

    # Exhaust the graph_retrieval limit (4) from other threads
    holders: list[ToolSlot] = []
    entered = threading.Event()

    def hold_slots() -> None:
        for _ in range(4):
            s = ToolSlot("graph_retrieval")
            s.__enter__()
            holders.append(s)
        entered.set()

    t = threading.Thread(target=hold_slots)
    t.start()
    entered.wait(timeout=5)
    t.join()

    # 5th acquire must queue (wait) rather than exceed the cap
    with ToolSlot("graph_retrieval") as slot:
        assert slot.status in {"waited", "timeout"}

    # Release everything
    for s in holders:
        s.__exit__(None, None, None)

    # Slot free again
    with ToolSlot("graph_retrieval") as after:
        assert after.status == "immediate"

    stats = get_contention_stats().get("graph_retrieval")
    assert stats is not None and stats["wait_events"] >= 1


def test_worker_fails_task_on_concurrency_timeout(monkeypatch) -> None:
    """A saturated tool must fail the task cleanly (backpressure), not hang."""
    from agent.nodes import concurrency
    from agent.nodes.parallel_nodes import sub_task_worker

    monkeypatch.setattr(concurrency, "MAX_WAIT_SECONDS", 0.2)

    # Occupy all 8 safe_math slots (TOOL_CONCURRENCY_LIMITS["safe_math"])
    limit = concurrency.TOOL_CONCURRENCY_LIMITS["safe_math"]
    holders = [concurrency.ToolSlot("safe_math") for _ in range(limit)]
    for h in holders:
        h.__enter__()
    try:
        out = sub_task_worker(
            {
                "task_id": "task_x",
                "target_tool": "safe_math",
                "payload": {"expression": "add(1, 2)"},
            }
        )
        env = out["sub_task_results"]["task_x"]
        assert env["success"] is False
        assert "Concurrency limit timeout" in env["error"]
    finally:
        for h in holders:
            h.__exit__(None, None, None)
