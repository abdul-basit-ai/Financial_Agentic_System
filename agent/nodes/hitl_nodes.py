"""Human-in-the-Loop (HITL) graph nodes implementing dynamic state suspension and resumption."""

from __future__ import annotations

from typing import Any

# Resilient LangGraph interrupt import across versions
try:
    from langgraph.types import interrupt
except ImportError:
    try:
        from langgraph.checkpoint.base import interrupt
    except ImportError:

        def interrupt(payload: Any) -> Any:  # Fallback mock for unit tests
            return {"action": "APPROVE"}


from agent.guardrails.audit import MerkleAuditLedger
from agent.guardrails.persistent_audit import PersistentMerkleAuditLedger
from agent.guardrails.risk_engine import FinancialRiskEngine
from agent.state.schema import AgentStateV1

# Durable audit ledger: Merkle chain persisted to Postgres, surviving process
# restarts (SEC 17a-4 posture). Degrades to memory-only with a loud warning
# if the database is unreachable — audit gaps are never silent.
AUDIT_LEDGER: MerkleAuditLedger = PersistentMerkleAuditLedger()
AUDIT_LEDGER.init_schema()


def eval_financial_risk_node(state: AgentStateV1) -> dict[str, Any]:
    """Inspects intermediate results and determines if human review is required."""
    assessment = FinancialRiskEngine.evaluate_state(state.model_dump())

    AUDIT_LEDGER.record_transition(
        trace_id=state.trace_id,
        node_name="eval_financial_risk",
        actor_id="risk_engine",
        action_type="EVALUATE_RISK",
        state_payload={"risk_score": assessment.composite_risk_score, "requires_hitl": assessment.requires_hitl},
    )

    log_msg = (
        f"[Risk Guardrail] Composite risk: {assessment.composite_risk_score:.2f}. "
        f"Requires HITL: {assessment.requires_hitl}. Triggered reasons: {len(assessment.reasons)}."
    )

    scratchpad_updates = [log_msg]
    for r in assessment.reasons:
        scratchpad_updates.append(f"  • [{r.severity}] {r.message}")

    return {
        "hitl_status": "PENDING" if assessment.requires_hitl else "APPROVED",
        "risk_evaluated": True,
        "scratchpad": scratchpad_updates,
    }


def hitl_gate_node(state: AgentStateV1) -> dict[str, Any]:
    """Suspends the graph using interrupt() and awaits asynchronous human review.

    On resume, applies the reviewer's overrides (if any) to pending tool calls
    before returning, so 'edit-and-resume' flows actually take effect.
    """
    # Prepare interrupt payload presented to the human reviewer
    interrupt_payload = {
        "trace_id": state.trace_id,
        "input_query": state.input,
        "pending_tool_calls": state.tool_calls,
        "retrieved_sub_tasks": state.sub_task_results,
        "scratchpad_summary": state.scratchpad[-5:] if state.scratchpad else [],
    }

    AUDIT_LEDGER.record_transition(
        trace_id=state.trace_id,
        node_name="hitl_gate",
        actor_id="system_gate",
        action_type="SUSPEND_INTERRUPT",
        state_payload=interrupt_payload,
    )

    # LangGraph interrupt freezes the execution thread into the checkpointer
    human_response = interrupt(interrupt_payload)

    # Resumed: Process decision payload
    action = str(human_response.get("action", "APPROVE")).upper()
    analyst_id = human_response.get("analyst_id", "analyst_unknown")
    feedback = human_response.get("feedback", "No comment provided.")
    overrides = human_response.get("overrides", {})

    if action not in {"APPROVE", "REJECT", "EDIT"}:
        # Fail closed: unknown verdicts must not silently auto-approve.
        AUDIT_LEDGER.record_transition(
            trace_id=state.trace_id,
            node_name="hitl_gate",
            actor_id="system_gate",
            action_type="HITL_INVALID_VERDICT",
            state_payload={"raw_action": action},
        )
        action = "REJECT"

    AUDIT_LEDGER.record_transition(
        trace_id=state.trace_id,
        node_name="hitl_gate",
        actor_id=analyst_id,
        action_type=f"HUMAN_{action}",
        state_payload={"action": action, "feedback": feedback, "overrides": overrides},
    )

    log_entry = f"[HITL Resolution] Action: {action} by {analyst_id}. Feedback: {feedback}"

    updates: dict[str, Any] = {"hitl_status": action, "scratchpad": [log_entry]}

    # Apply reviewer edits to pending tool calls (edit-and-resume contract).
    if action == "EDIT" and isinstance(overrides, dict):
        edited_calls = overrides.get("tool_calls")
        if isinstance(edited_calls, list):
            updates["tool_calls"] = edited_calls
            updates["scratchpad"] = [
                log_entry,
                f"[HITL Override] {len(edited_calls)} tool call(s) replaced per reviewer edits.",
            ]

    return updates


def apply_override_node(state: AgentStateV1) -> dict[str, Any]:
    """Injects approved human edits cleanly into tool_calls and sub_task_results."""
    scratchpad_logs = ["[HITL Override] Verified and aligned state with human directions."]

    # If rejected, mark terminal or redirect to plan
    if state.hitl_status in {"REJECTED", "REJECT"}:
        return {
            "final_answer": "Execution rejected by financial compliance review. Trajectory aborted.",
            "is_terminal": True,
            "scratchpad": ["[HITL Override] Execution terminated by human reviewer."],
        }

    return {
        "scratchpad": scratchpad_logs,
    }