from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent.guardrails.persistent_audit import PersistentMerkleAuditLedger


def record(ledger, number, event_id=None):
    return ledger.record_transition(
        trace_id="trace",
        node_name="node",
        actor_id="reviewer",
        action_type="REVIEW",
        state_payload={"number": number},
        metadata={"event_id": event_id} if event_id else {},
    )


def test_chain_survives_restart_after_multiple_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")
    path = str(tmp_path / "audit.sqlite3")
    first = PersistentMerkleAuditLedger(path=path)
    assert first.init_schema()
    entries = [record(first, number) for number in range(5)]
    restarted = PersistentMerkleAuditLedger(path=path)
    assert restarted.init_schema()
    next_entry = record(restarted, 5)
    assert next_entry.entry_index == 5
    assert next_entry.prev_hash == entries[-1].entry_hash
    assert restarted.verify_integrity()[0]


def test_parallel_writers_do_not_lose_audit_events(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")
    path = str(tmp_path / "audit.sqlite3")
    assert PersistentMerkleAuditLedger(path=path).init_schema()

    def write(number):
        return record(PersistentMerkleAuditLedger(path=path), number).entry_index

    with ThreadPoolExecutor(max_workers=6) as pool:
        indices = list(pool.map(write, range(18)))
    assert sorted(indices) == list(range(18))
    assert PersistentMerkleAuditLedger(path=path).verify_integrity()[0]


def test_replayed_interrupt_audit_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")
    path = str(tmp_path / "audit.sqlite3")
    first = PersistentMerkleAuditLedger(path=path)
    one = record(first, 1, "suspend:trace:0")
    two = record(PersistentMerkleAuditLedger(path=path), 1, "suspend:trace:0")
    assert one == two
    third = record(first, 2)
    assert third.entry_index == 1
    assert first.verify_integrity()[0]


def test_audit_metadata_tampering_is_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")
    path = str(tmp_path / "audit.sqlite3")
    ledger = PersistentMerkleAuditLedger(path=path)
    record(ledger, 1, "event-1")
    with sqlite3.connect(path) as connection:
        payload = json.loads(
            connection.execute(
                "SELECT payload FROM audit_events WHERE entry_index=0"
            ).fetchone()[0]
        )
        payload["metadata"]["event_id"] = "forged"
        connection.execute(
            "UPDATE audit_events SET payload=? WHERE entry_index=0",
            (json.dumps(payload),),
        )
    assert ledger.verify_integrity()[0] is False


def test_audit_unavailable_fails_without_memory_only_success(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")
    blocker = tmp_path / "blocker"
    blocker.write_text("file")
    ledger = PersistentMerkleAuditLedger(path=str(blocker / "audit.sqlite3"))
    assert ledger.init_schema() is False
    with pytest.raises(RuntimeError, match="unavailable"):
        record(ledger, 1)
    assert ledger.entries == []
