"""Integration test suite for Phase 13 Production Serving & Gateway Architecture.

Verifies:
1. Health check and distributed tracing headers.
2. Real-time Server-Sent Events (SSE) streaming frame delivery.
3. Asynchronous decoupled job submission and status polling.
4. Thread state inspection.
5. Human-in-the-Loop (HITL) resumption endpoint.
"""

from __future__ import annotations

import asyncio
import uuid
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from agent.graph import create_financial_agent
from agent.server.app import create_app


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    # Use MemorySaver for deterministic isolated testing
    app.state.agent = create_financial_agent(checkpointer=MemorySaver())
    return TestClient(app)


# =====================================================================
# 1. Gateway Health & Middleware Tests
# =====================================================================


def test_health_check_and_headers(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "X-Trace-Id" in response.headers
    assert "X-Response-Time-Ms" in response.headers


# =====================================================================
# 2. Server-Sent Events (SSE) Streaming Tests
# =====================================================================


def test_sse_query_stream(client: TestClient) -> None:
    payload = {
        "query": "What was Amazon's revenue in 2020?",
        "company_identifier": "AMZN",
        "thread_id": f"sse_test_{uuid.uuid4().hex[:8]}",
    }

    with client.stream("POST", "/api/v1/query/stream", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        events_received = []
        for line in response.iter_lines():
            if line.startswith("event: "):
                events_received.append(line.replace("event: ", "").strip())

        # Verify stream structure
        assert "lifecycle" in events_received
        assert "node_update" in events_received
        assert any(e in {"final_answer", "complete", "interrupt"} for e in events_received)


# =====================================================================
# 3. Asynchronous Jobs & Polling Tests
# =====================================================================


def test_async_job_submission_and_polling(client: TestClient) -> None:
    thread_id = f"async_test_{uuid.uuid4().hex[:8]}"
    payload = {
        "query": "What was Microsoft's revenue in 2021?",
        "company_identifier": "MSFT",
        "thread_id": thread_id,
    }

    # 1. Submit async query
    submit_res = client.post("/api/v1/query/async", json=payload)
    assert submit_res.status_code == 202
    job_info = submit_res.json()
    assert "job_id" in job_info
    job_id = job_info["job_id"]

    # 2. Poll for completion
    poll_res = client.get(f"/api/v1/jobs/{job_id}")
    assert poll_res.status_code == 200
    status_data = poll_res.json()
    assert status_data["job_id"] == job_id
    assert status_data["status"] in {"QUEUED", "RUNNING", "COMPLETED", "SUSPENDED_HITL"}


# =====================================================================
# 4. State Inspection & HITL Resumption Tests
# =====================================================================


def test_thread_state_inspection(client: TestClient) -> None:
    thread_id = f"state_test_{uuid.uuid4().hex[:8]}"

    # Query stream to initialize state in memory
    client.post(
        "/api/v1/query/stream",
        json={"query": "Apple operating income 2020", "thread_id": thread_id},
    )

    # Inspect state
    res = client.get(f"/api/v1/threads/{thread_id}/state")
    assert res.status_code == 200
    state_data = res.json()
    assert state_data["thread_id"] == thread_id
    assert "state_values" in state_data


def test_resume_non_paused_thread_returns_400(client: TestClient) -> None:
    thread_id = f"resume_err_{uuid.uuid4().hex[:8]}"

    # Run query to completion
    client.post(
        "/api/v1/query/stream",
        json={"query": "Quick query", "thread_id": thread_id},
    )

    # Resuming an unpaused thread should return 400 Bad Request
    resume_payload = {
        "action": "APPROVE",
        "analyst_id": "analyst_1",
        "feedback": "LGTM",
    }
    res = client.post(f"/api/v1/threads/{thread_id}/resume", json=resume_payload)
    assert res.status_code == 400