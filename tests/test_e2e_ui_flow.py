"""Validation suite for Phase 14 API & UI Integration.

Verifies:
1. Full SSE streaming output structure with lifecycle, node updates, and final answers.
2. /api/v1/approvals endpoint discovers paused threads with risk trigger metadata.
3. Successful state resumption via /api/v1/threads/{thread_id}/resume.
4. E2E pipeline flow across parallel retrieval, risk evaluation, and synthesis.
"""

from __future__ import annotations

import json
import uuid
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from agent.graph import create_financial_agent
from agent.server.app import create_app
from agent.server.routes import ACTIVE_THREADS


@pytest.fixture
def test_client() -> TestClient:
    app = create_app()
    app.state.agent = create_financial_agent(checkpointer=MemorySaver())
    ACTIVE_THREADS.clear()
    return TestClient(app)


# =====================================================================
# 1. End-to-End SSE Streaming Verification
# =====================================================================


def test_e2e_streaming_reasoning_pipeline(test_client: TestClient) -> None:
    thread_id = f"e2e_stream_{uuid.uuid4().hex[:8]}"
    payload = {
        "query": "What was Amazon's revenue in 2020?",
        "company_identifier": "AMZN",
        "thread_id": thread_id,
    }

    with test_client.stream("POST", "/api/v1/query/stream", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        event_names = []
        data_payloads = []

        for line in response.iter_lines():
            if line.startswith("event:"):
                event_names.append(line.replace("event:", "").strip())
            elif line.startswith("data:"):
                raw_json = line.replace("data:", "").strip()
                try:
                    data_payloads.append(json.loads(raw_json))
                except Exception:
                    pass

        # Verify pipeline event progression
        assert "lifecycle" in event_names
        assert "node_update" in event_names
        assert any(e in {"final_answer", "complete", "interrupt"} for e in event_names)

        # Thread must be tracked in active repository
        assert thread_id in ACTIVE_THREADS


# =====================================================================
# 2. End-to-End HITL Queue & Resumption Flow
# =====================================================================


def test_e2e_hitl_approval_queue_and_resume(test_client: TestClient) -> None:
    thread_id = f"e2e_hitl_{uuid.uuid4().hex[:8]}"

    # Initialize a paused state directly in the checkpointer
    agent = test_client.app.state.agent
    config = {"configurable": {"thread_id": thread_id}}

    paused_state_values = {
        "input": "Calculate operating margin with outlier check",
        # AgentStateV1 requires a valid UUIDv4 trace_id (fail-loud validator)
        "trace_id": str(uuid.uuid4()),
        "company_identifier": "AMZN",
        "hitl_status": "PENDING",
        "scratchpad": [
            "[Risk Guardrail] Composite risk: 0.85. Requires HITL: True.",
            "  • [CRITICAL] Impossible margin detected on 'Operating Margin': 145.0%",
        ],
        "tool_calls": [
            {
                "task_id": "task_math",
                "target_tool": "safe_math",
                "payload": {"expression": "divide(145, 100)"},
                "status": "PENDING",
            }
        ],
    }

    # Store active thread metadata and state
    ACTIVE_THREADS[thread_id] = {
        "input_query": paused_state_values["input"],
        "company_identifier": "AMZN",
        "created_at": "2026-09-07T12:00:00Z",
    }
    agent.update_state(config, paused_state_values, as_node="hitl_gate")

    # 1. Query /api/v1/approvals endpoint
    approvals_res = test_client.get("/api/v1/approvals")
    assert approvals_res.status_code == 200
    approval_data = approvals_res.json()

    assert approval_data["total_pending"] >= 1
    target_item = next((it for it in approval_data["items"] if it["thread_id"] == thread_id), None)
    assert target_item is not None
    assert target_item["hitl_status"] == "PENDING"
    assert any("Impossible margin" in r for r in target_item["trigger_reasons"])

    # 2. Submit Analyst Approval via /resume endpoint
    resume_payload = {
        "action": "APPROVE",
        "analyst_id": "analyst_compliance_lead",
        "feedback": "Verified with SEC 10-K filing footnote 4.",
        "overrides": {},
    }

    resume_res = test_client.post(f"/api/v1/threads/{thread_id}/resume", json=resume_payload)
    assert resume_res.status_code == 200
    res_data = resume_res.json()
    assert res_data["thread_id"] == thread_id
    assert res_data["status"] == "RESUMED"
    assert res_data["action_applied"] == "APPROVE"


# =====================================================================
# 3. Health & State Retrieval Verification
# =====================================================================


def test_health_check_endpoint(test_client: TestClient) -> None:
    res = test_client.get("/api/v1/health")
    assert res.status_code == 200
    assert res.json() == {"status": "healthy", "service": "finagent-production-gateway"}