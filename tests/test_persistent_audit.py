"""Tests for the Postgres-backed persistent Merkle audit ledger.

Verifies:
1. Schema creation and durable writes.
2. Chain continuity across a simulated process restart (fresh ledger instance
   continues from the DB tail, no fork at genesis).
3. Tamper detection on the DURABLE chain (verify_integrity(from_db=True)).
4. Graceful, loud degradation to memory-only when Postgres is unreachable.
"""

from __future__ import annotations

import os

import pytest

from agent.guardrails.audit import MerkleAuditLedger
from agent.guardrails.persistent_audit import PersistentMerkleAuditLedger


@pytest.fixture()
def pg_ledger():
    ledger = PersistentMerkleAuditLedger()
    ok = ledger.init_schema()
    if not ok:
        pytest.skip("Postgres unavailable for persistent audit tests")
    yield ledger
    ledger.close()


@pytest.fixture()
def clean_table(pg_ledger):
    with pg_ledger._get_connection().cursor() as cur:
        cur.execute("DELETE FROM audit_ledger")
    pg_ledger._get_connection().commit()
    pg_ledger.entries = []
    yield pg_ledger
    with pg_ledger._get_connection().cursor() as cur:
        cur.execute("DELETE FROM audit_ledger")
    pg_ledger._get_connection().commit()


def test_persistent_writes_are_durable(clean_table) -> None:
    ledger = clean_table
    ledger.record_transition(
        trace_id="t1", node_name="plan", actor_id="system",
        action_type="CREATE_PLAN", state_payload={"step": 1},
    )
    ledger.record_transition(
        trace_id="t1", node_name="eval_risk", actor_id="risk_engine",
        action_type="EVALUATE_RISK", state_payload={"risk": 0.2},
    )

    # Fresh instance (simulated restart) reads the same durable chain
    fresh = PersistentMerkleAuditLedger()
    fresh.init_schema()
    tail = fresh._load_tail()
    assert tail is not None
    assert tail.action_type == "EVALUATE_RISK"
    assert tail.entry_hash == ledger.entries[-1].entry_hash
    fresh.close()


def test_chain_continues_across_restart_without_fork(clean_table) -> None:
    first = clean_table
    e1 = first.record_transition(
        trace_id="t1", node_name="a", actor_id="s",
        action_type="ACT_1", state_payload={"n": 1},
    )

    # Simulate restart: brand-new ledger instance with empty memory
    second = PersistentMerkleAuditLedger()
    second.init_schema()
    e2 = second.record_transition(
        trace_id="t1", node_name="b", actor_id="s",
        action_type="ACT_2", state_payload={"n": 2},
    )

    # The restart entry must chain onto the DB tail, not genesis
    assert e2.prev_hash == e1.entry_hash
    assert e2.entry_index == e1.entry_index + 1

    # Durable verification spans both processes' entries
    ok, msg = second.verify_integrity(from_db=True)
    assert ok is True
    assert "Zero discrepancies" in msg
    second.close()


def test_db_tampering_is_detected(clean_table) -> None:
    ledger = clean_table
    for i in range(3):
        ledger.record_transition(
            trace_id="t1", node_name=f"n{i}", actor_id="s",
            action_type=f"ACT_{i}", state_payload={"i": i},
        )

    ok, _ = ledger.verify_integrity(from_db=True)
    assert ok is True

    # Tamper DIRECTLY in the database (simulates DB-level edit)
    with ledger._get_connection().cursor() as cur:
        cur.execute(
            "UPDATE audit_ledger SET actor_id = 'attacker' WHERE entry_index = 1"
        )
    ledger._get_connection().commit()

    ok_tampered, msg = ledger.verify_integrity(from_db=True)
    assert ok_tampered is False
    assert "Tampered" in msg or "recomputation" in msg

    # Restore for fixture cleanup
    with ledger._get_connection().cursor() as cur:
        cur.execute("DELETE FROM audit_ledger WHERE entry_index >= 1")
        cur.execute(
            "INSERT INTO audit_ledger (entry_index, timestamp_utc, trace_id, node_name,"
            " actor_id, action_type, state_hash, prev_hash, entry_hash, metadata)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                ledger.entries[1].entry_index, ledger.entries[1].timestamp_utc,
                ledger.entries[1].trace_id, ledger.entries[1].node_name,
                ledger.entries[1].actor_id, ledger.entries[1].action_type,
                ledger.entries[1].state_hash, ledger.entries[1].prev_hash,
                ledger.entries[1].entry_hash, "{}",
            ),
        )
    ledger._get_connection().commit()


def test_degraded_mode_is_memory_only_and_loud(clean_table, capsys) -> None:
    # Point at an unreachable server -> init_schema must fail OPEN with a warning
    degraded = PersistentMerkleAuditLedger(
        host="localhost", port=9999, dbname="financial_agent",
        user="postgres", password="password",
    )
    ok = degraded.init_schema()
    assert ok is False
    captured = capsys.readouterr()
    assert "MEMORY-ONLY" in captured.out

    # Recording still works in-memory so agent execution is not blocked
    entry = degraded.record_transition(
        trace_id="t1", node_name="a", actor_id="s",
        action_type="ACT", state_payload={"x": 1},
    )
    assert entry.entry_index == 0
    ok_mem, _ = degraded.verify_integrity()
    assert ok_mem is True


def test_hitl_nodes_use_persistent_ledger() -> None:
    """The graph's AUDIT_LEDGER must be the persistent implementation."""
    from agent.nodes.hitl_nodes import AUDIT_LEDGER

    assert isinstance(AUDIT_LEDGER, PersistentMerkleAuditLedger)
    # And it must be a strict subtype of the base contract
    assert isinstance(AUDIT_LEDGER, MerkleAuditLedger)
