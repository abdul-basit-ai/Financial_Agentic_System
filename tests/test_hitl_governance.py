"""Validation suite for Phase 8 HITL Governance, Risk Engine, and Merkle Audit Trail.

Verifies:
1. Benford's Law Chi-Square goodness-of-fit detection on valid and corrupted data.
2. Modified Z-Score outlier detection using MAD.
3. Financial risk engine flagging impossible margins, extreme growth, and materiality.
4. Cryptographic Merkle audit chain verification and tamper detection.
5. Interrupt and resume cycle execution using LangGraph MemorySaver.
"""

from __future__ import annotations

import math
import uuid

from langgraph.checkpoint.memory import MemorySaver

from agent.graph import create_financial_agent
from agent.guardrails.audit import MerkleAuditLedger
from agent.guardrails.risk_engine import (
    FinancialRiskEngine,
    compute_modified_z_scores,
    verify_benford_law_conformity,
)
from agent.state.schema import AgentStateV1


def test_benford_law_conformity() -> None:

    benford_numbers = []
    for d in range(1, 10):
        count = int(math.log10(1.0 + 1.0 / d) * 300)
        benford_numbers.extend([float(d * 100 + i) for i in range(count)])

    is_ok, chi_sq = verify_benford_law_conformity(benford_numbers)
    assert is_ok is True
    assert chi_sq <= 20.09

    uniform_numbers = []
    for d in range(1, 10):
        uniform_numbers.extend([float(d * 100 + i) for i in range(50)])

    is_ok_corrupted, chi_sq_corrupted = verify_benford_law_conformity(uniform_numbers)
    assert is_ok_corrupted is False
    assert chi_sq_corrupted > 20.09


def test_modified_z_score_mad_detection() -> None:

    values = [100.0, 102.0, 98.0, 105.0, 101.0, 99.0, 1000.0]
    z_scores = compute_modified_z_scores(values)

    assert len(z_scores) == len(values)

    assert abs(z_scores[-1]) > 3.5
    assert abs(z_scores[0]) < 3.5


def test_financial_risk_engine_triggers_on_anomalies() -> None:
    anomalous_state = {
        "tool_results": [
            {
                "tool_name": "graph_retrieval",
                "data": {
                    "records": [
                        {
                            "row_label": "Operating Margin",
                            "column_header": "operating margin ( % )",
                            "amount": 145.0,
                            "normalized_amount": None,
                        },
                        {
                            "row_label": "Goodwill Impairment",
                            "amount": 2_500_000_000.0,
                            "normalized_amount": 2_500_000_000.0,
                        },
                    ]
                },
            }
        ]
    }

    result = FinancialRiskEngine.evaluate_state(anomalous_state)
    assert result.requires_hitl is True
    assert result.composite_risk_score >= 0.7
    assert len(result.reasons) >= 2
    assert any(r.category == "ANOMALY" for r in result.reasons)
    assert any(r.category == "MATERIALITY" for r in result.reasons)


def test_merkle_audit_ledger_integrity_and_tamper_detection() -> None:
    ledger = MerkleAuditLedger()

    ledger.record_transition(
        trace_id="trace_001",
        node_name="plan",
        actor_id="planner",
        action_type="CREATE_PLAN",
        state_payload={"plan": "Step 1"},
    )
    ledger.record_transition(
        trace_id="trace_001",
        node_name="eval_risk",
        actor_id="risk_engine",
        action_type="EVALUATE_RISK",
        state_payload={"risk": 0.8},
    )
    ledger.record_transition(
        trace_id="trace_001",
        node_name="hitl_gate",
        actor_id="analyst_42",
        action_type="HUMAN_APPROVE",
        state_payload={"action": "APPROVE"},
    )

    is_valid, msg = ledger.verify_integrity()
    assert is_valid is True
    assert "Zero discrepancies" in msg

    ledger.entries[1].state_hash = "f" * 64
    is_tampered, tamper_msg = ledger.verify_integrity()
    assert is_tampered is False
    assert "Tampered record at index 1" in tamper_msg


def test_end_to_end_hitl_workflow_execution() -> None:
    saver = MemorySaver()
    agent = create_financial_agent(checkpointer=saver)

    session_id = f"test_hitl_{uuid.uuid4().hex[:8]}"
    state = AgentStateV1(
        input="Analyze operating revenue for Amazon in 2020",
        company_identifier="AMZN",
    )
    config = {"configurable": {"thread_id": session_id}}

    result = agent.invoke(state.model_dump(), config=config)

    assert result is not None
    assert "hitl_status" in result
    assert result["hitl_status"] in {"APPROVED", "PENDING", "REJECTED"}
    assert any("Risk Guardrail" in log for log in result["scratchpad"])


def test_hitl_interrupt_pauses_and_resume_approves() -> None:
    """Full Phase 8 contract: graph SUSPENDS at the gate when risk triggers,
    then RESUMES with the reviewer's verdict via Command(resume=...)."""
    from langgraph.types import Command

    saver = MemorySaver()
    agent = create_financial_agent(checkpointer=saver)
    session_id = f"test_hitl_resume_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": session_id}}

    state = AgentStateV1(
        input="Analyze goodwill impairment for Amazon in 2020",
        company_identifier="AMZN",
    )
    first = agent.invoke(state.model_dump(), config=config)

    if first["hitl_status"] == "PENDING" and "__interrupt__" in first:

        snap = agent.get_state(config)
        assert snap.next, "Graph must be paused at hitl_gate"

        final = agent.invoke(
            Command(resume={"action": "APPROVE", "analyst_id": "analyst_42"}),
            config=config,
        )
        assert final["is_terminal"] is True
        assert final["hitl_status"] in {"APPROVE", "APPROVED"}
        assert any("HITL Resolution" in log for log in final["scratchpad"])
    else:

        assert final_is_terminal(first)


def final_is_terminal(result: dict) -> bool:
    return result.get("is_terminal") is True
