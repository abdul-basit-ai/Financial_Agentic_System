"""Human-in-the-Loop (HITL) graph nodes implementing dynamic state suspension and resumption."""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from agent.guardrails.persistent_audit import PersistentMerkleAuditLedger
from agent.guardrails.risk_engine import FinancialRiskEngine
from agent.state.schema import AgentStateV1

AUDIT_LEDGER: PersistentMerkleAuditLedger = PersistentMerkleAuditLedger()
_schema_initialized = False


def _ensure_audit_schema() -> None:
    """Initializes the audit table once per process (idempotent DDL)."""
    global _schema_initialized
    if not _schema_initialized:
        if not AUDIT_LEDGER.init_schema():
            raise RuntimeError("Durable audit storage is unavailable.")
        _schema_initialized = True


def eval_financial_risk_node(state: AgentStateV1) -> dict[str, Any]:
    """Inspects intermediate results and determines if human review is required."""
    _ensure_audit_schema()
    assessment = FinancialRiskEngine.evaluate_state(state.model_dump())

    AUDIT_LEDGER.record_transition(
        trace_id=state.trace_id,
        node_name="eval_financial_risk",
        actor_id="risk_engine",
        action_type="EVALUATE_RISK",
        state_payload={
            "risk_score": assessment.composite_risk_score,
            "requires_hitl": assessment.requires_hitl,
        },
        metadata={"event_id": f"risk:{state.trace_id}:{state.iteration_count}"},
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
    _ensure_audit_schema()

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
        metadata={"event_id": f"suspend:{state.trace_id}:{state.iteration_count}"},
    )

    human_response = interrupt(interrupt_payload)

    action = str(human_response.get("action", "REJECT")).upper()
    analyst_id = human_response.get("analyst_id", "analyst_unknown")
    feedback = human_response.get("feedback", "No comment provided.")
    overrides = human_response.get("overrides", {})

    if action not in {"APPROVE", "REJECT", "EDIT"}:

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

    log_entry = (
        f"[HITL Resolution] Action: {action} by {analyst_id}. Feedback: {feedback}"
    )

    updates: dict[str, Any] = {"hitl_status": action, "scratchpad": [log_entry]}

    if action == "EDIT" and isinstance(overrides, dict):
        edited_calls = overrides.get("tool_calls")
        if isinstance(edited_calls, list) and edited_calls:
            from agent.nodes.plan_node import rekey_calls, validate_tool_payload

            seen: set[str] = set()
            for call in edited_calls:
                if (
                    not isinstance(call, dict)
                    or not call.get("task_id")
                    or call["task_id"] in seen
                ):
                    raise ValueError("Edited calls require unique task IDs.")
                if not set(call.get("dependencies", [])).issubset(seen):
                    raise ValueError(
                        "Edited calls must form a self-contained ordered plan."
                    )
                validate_tool_payload(call["target_tool"], call.get("payload", {}))
                seen.add(call["task_id"])
            updates["tool_calls"] = rekey_calls(edited_calls, state.iteration_count)
            updates["iteration_count"] = state.iteration_count + 1
            updates["risk_evaluated"] = False
            updates["citations"] = []
            updates["final_answer"] = None
            updates["scratchpad"] = [
                log_entry,
                f"[HITL Override] {len(edited_calls)} tool call(s) replaced per reviewer edits.",
            ]
        else:
            raise ValueError("An edited plan must contain at least one tool call.")

    return updates


def apply_override_node(state: AgentStateV1) -> dict[str, Any]:
    """Injects approved human edits cleanly into tool_calls and sub_task_results."""
    scratchpad_logs = [
        "[HITL Override] Verified and aligned state with human directions."
    ]

    if state.hitl_status in {"REJECTED", "REJECT"}:
        return {
            "final_answer": "Execution rejected by financial compliance review. Trajectory aborted.",
            "is_terminal": True,
            "answer_status": "rejected",
            "citations": [],
            "scratchpad": ["[HITL Override] Execution terminated by human reviewer."],
        }

    return {
        "scratchpad": scratchpad_logs,
    }
