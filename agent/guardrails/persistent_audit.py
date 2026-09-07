"""Persistent Merkle-chain audit ledger backed by PostgreSQL.

Extends the in-process MerkleAuditLedger with durable storage: every entry is
written to Postgres inside the same hash chain, so the chain survives process
restarts and remains tamper-evident across deployments (SEC 17a-4 posture).

Integrity model:
- entry_hash = SHA-256(prev_hash | trace | node | actor | action | state_hash | ts)
- The DB stores the full chain; verification recomputes hashes in index order.
- Writes are INSERT-only; UPDATE/DELETE on the table is the tamper vector the
  chain detects — no database privilege should grant those to the app role in
  production (see README note in guardrails).

Fail-open policy: if Postgres is unreachable the caller may fall back to the
in-memory ledger so agent execution continues; audit gaps are announced loudly
rather than silently (compliance teams must know when the chain is degraded).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

from agent.guardrails.audit import AuditEntry, MerkleAuditLedger

AUDIT_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS audit_ledger (
    entry_index BIGINT PRIMARY KEY,
    timestamp_utc TIMESTAMPTZ NOT NULL,
    trace_id TEXT NOT NULL,
    node_name TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    state_hash CHAR(64) NOT NULL,
    prev_hash CHAR(64) NOT NULL,
    entry_hash CHAR(64) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_audit_trace ON audit_ledger (trace_id);
CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_ledger (timestamp_utc);
"""


class PersistentMerkleAuditLedger(MerkleAuditLedger):
    """Merkle-chain ledger durable across process restarts via Postgres.

    Chain continuity is maintained against the LAST entry in the database,
    not just in-memory entries, so a restarted process continues the same
    hash chain (no fork at the genesis boundary).
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        dbname: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        super().__init__()
        self.conn_params = {
            "host": host or os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(port or os.getenv("POSTGRES_PORT", "5432")),
            "dbname": dbname or os.getenv("POSTGRES_DB", "financial_agent"),
            "user": user or os.getenv("POSTGRES_USER", "postgres"),
            "password": password or os.getenv("POSTGRES_PASSWORD", "password"),
        }
        self._conn: Any = None
        self._db_available = False

    def _get_connection(self) -> Any:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(**self.conn_params)
        return self._conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
            self._conn = None

    def init_schema(self) -> bool:
        """Creates the audit table. Returns False (with loud warning) if DB is
        unreachable — caller decides whether to degrade to memory-only."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(AUDIT_SCHEMA_DDL)
            conn.commit()
            self._db_available = True
            return True
        except Exception as exc:
            print(
                f"[audit] WARNING: Postgres unreachable ({type(exc).__name__}: {exc}). "
                f"Audit chain is MEMORY-ONLY for this process — entries will NOT "
                f"survive restart and cross-process verification is unavailable.",
                flush=True,
            )
            self._db_available = False
            return False

    def _load_tail(self) -> AuditEntry | None:
        """Loads the highest-index entry from the DB to continue the chain."""
        if not self._db_available:
            return None
        try:
            conn = self._get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM audit_ledger ORDER BY entry_index DESC LIMIT 1"
                )
                row = cur.fetchone()
            if not row:
                return None
            return AuditEntry(
                entry_index=int(row["entry_index"]),
                timestamp_utc=row["timestamp_utc"].isoformat(),
                trace_id=row["trace_id"],
                node_name=row["node_name"],
                actor_id=row["actor_id"],
                action_type=row["action_type"],
                state_hash=row["state_hash"],
                prev_hash=row["prev_hash"],
                entry_hash=row["entry_hash"],
                metadata=row["metadata"] if isinstance(row["metadata"], dict) else {},
            )
        except Exception:
            return None

    def record_transition(
        self,
        trace_id: str,
        node_name: str,
        actor_id: str,
        action_type: str,
        state_payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Records a transition in memory AND persists to Postgres.

        The chain continues from the DB tail when this process is fresh, so
        restarts never fork the hash chain.
        """
        # Continue from DB tail if our in-memory list is empty/restarting
        if not self.entries and self._db_available:
            tail = self._load_tail()
            if tail is not None:
                self.entries.append(tail)

        entry = super().record_transition(
            trace_id=trace_id,
            node_name=node_name,
            actor_id=actor_id,
            action_type=action_type,
            state_payload=state_payload,
            metadata=metadata,
        )

        if self._db_available:
            try:
                conn = self._get_connection()
                with conn.cursor() as cur:
                    # Skip re-writing a tail entry loaded from the DB itself
                    cur.execute(
                        """
                        INSERT INTO audit_ledger (
                            entry_index, timestamp_utc, trace_id, node_name,
                            actor_id, action_type, state_hash, prev_hash,
                            entry_hash, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (entry_index) DO NOTHING
                        """,
                        (
                            entry.entry_index,
                            entry.timestamp_utc,
                            entry.trace_id,
                            entry.node_name,
                            entry.actor_id,
                            entry.action_type,
                            entry.state_hash,
                            entry.prev_hash,
                            entry.entry_hash,
                            json.dumps(entry.metadata),
                        ),
                    )
                conn.commit()
            except Exception as exc:
                # Degrade loudly: memory chain continues, DB gap is announced.
                print(
                    f"[audit] WARNING: failed to persist audit entry #{entry.entry_index} "
                    f"({type(exc).__name__}: {exc}). Chain continues in memory only; "
                    f"DB will have a gap at this index.",
                    flush=True,
                )
                try:
                    self._conn.rollback()
                except Exception:
                    pass

        return entry

    def verify_integrity(self, from_db: bool = False) -> tuple[bool, str]:
        """Verifies the chain. from_db=True loads ALL entries from Postgres and
        verifies the durable chain (cross-restart, cross-process) instead of
        the in-memory view."""
        if from_db:
            if not self._db_available:
                return False, "Database unavailable; cannot verify durable chain."
            try:
                conn = self._get_connection()
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT * FROM audit_ledger ORDER BY entry_index ASC")
                    rows = cur.fetchall()
                db_entries = [
                    AuditEntry(
                        entry_index=int(r["entry_index"]),
                        timestamp_utc=r["timestamp_utc"].isoformat(),
                        trace_id=r["trace_id"],
                        node_name=r["node_name"],
                        actor_id=r["actor_id"],
                        action_type=r["action_type"],
                        state_hash=r["state_hash"],
                        prev_hash=r["prev_hash"],
                        entry_hash=r["entry_hash"],
                        metadata=r["metadata"] if isinstance(r["metadata"], dict) else {},
                    )
                    for r in rows
                ]
            except Exception as exc:
                return False, f"Failed to load durable chain: {exc}"

            saved, self.entries = self.entries, db_entries
            try:
                return super().verify_integrity()
            finally:
                self.entries = saved

        return super().verify_integrity()
