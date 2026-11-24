from __future__ import annotations

import json
import time

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes.hitl_nodes import apply_override_node, hitl_gate_node
from agent.state.schema import AgentStateV1


def test_e2e_streaming_reasoning_pipeline(api_client):
    response = api_client.post(
        "/api/v1/query/stream", json={"query": "Revenue?", "thread_id": "stream-demo"}
    )
    assert response.status_code == 200
    events = [
        line.removeprefix("event: ")
        for line in response.text.splitlines()
        if line.startswith("event: ")
    ]
    assert "lifecycle" in events
    assert "node_update" in events
    assert "final_answer" in events
    assert "complete" in events
    state = api_client.get("/api/v1/threads/stream-demo/state").json()
    assert state["state_values"]["is_terminal"] is True


def test_e2e_hitl_approval_queue_and_resume(api_client):
    graph = StateGraph(AgentStateV1)
    graph.add_node(
        "review_start",
        lambda state: {
            "hitl_status": "PENDING",
            "scratchpad": ["[CRITICAL] Evidence needs review"],
        },
    )
    graph.add_node("hitl_gate", hitl_gate_node)
    graph.add_node("apply_override", apply_override_node)
    graph.add_node(
        "finish",
        lambda state: {"final_answer": "Reviewed evidence.", "is_terminal": True},
    )
    graph.add_edge(START, "review_start")
    graph.add_edge("review_start", "hitl_gate")
    graph.add_edge("hitl_gate", "apply_override")
    graph.add_edge("apply_override", "finish")
    graph.add_edge("finish", END)
    agent = graph.compile(checkpointer=MemorySaver())
    api_client.app.state.agent = agent
    api_client.app.state.worker.agent = agent
    response = api_client.post(
        "/api/v1/query/async",
        json={"query": "Review the cited evidence", "thread_id": "review-demo"},
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    for _ in range(100):
        job = api_client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] != "QUEUED" and job["status"] != "RUNNING":
            break
        time.sleep(0.05)
    assert job["status"] == "SUSPENDED_HITL"
    queue = api_client.get("/api/v1/approvals").json()
    assert queue["total_pending"] == 1
    assert queue["items"][0]["thread_id"] == "review-demo"
    resumed = api_client.post(
        "/api/v1/threads/review-demo/resume",
        json={"action": "APPROVE", "analyst_id": "forged-id"},
    )
    assert resumed.status_code == 200
    assert resumed.json()["final_answer"] == "Reviewed evidence."
    state = api_client.get("/api/v1/threads/review-demo/state").json()
    assert state["is_paused"] is False
    assert api_client.get("/api/v1/approvals").json()["total_pending"] == 0


def test_stream_packets_are_json(api_client):
    response = api_client.post(
        "/api/v1/query/stream", json={"query": "Evidence?", "thread_id": "packet-demo"}
    )
    for line in response.text.splitlines():
        if line.startswith("data: "):
            assert isinstance(json.loads(line.removeprefix("data: ")), dict)


def test_health_check_endpoint(api_client):
    response = api_client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
