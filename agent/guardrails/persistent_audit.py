from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import Json

from agent.guardrails.audit import AuditEntry, MerkleAuditLedger

AUDIT_SCHEMA_DDL = "CREATE TABLE IF NOT EXISTS finagent.audit_events (entry_index BIGINT PRIMARY KEY, event_id TEXT UNIQUE, payload JSONB NOT NULL)"


class PersistentMerkleAuditLedger(MerkleAuditLedger):
    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        dbname: str | None = None,
        user: str | None = None,
        password: str | None = None,
        path: str | None = None,
    ) -> None:
        super().__init__()
        self.database_url = os.getenv("DATABASE_URL", "")
        self.conn_params = {
            "host": host,
            "port": port or 5432,
            "dbname": dbname,
            "user": user,
            "password": password,
            "connect_timeout": 3,
        }
        self.use_postgres = bool(self.database_url or host)
        self.path = path or os.getenv(
            "FINAGENT_AUDIT_PATH", "data/runtime/audit.sqlite3"
        )
        self._db_available = False

    @contextmanager
    def _connection(self):
        if self.use_postgres:
            conn = (
                psycopg2.connect(self.database_url, connect_timeout=5)
                if self.database_url
                else psycopg2.connect(**self.conn_params)
            )
        else:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, timeout=10)
        try:
            yield conn
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @property
    def table(self) -> str:
        return "finagent.audit_events" if self.use_postgres else "audit_events"

    def init_schema(self) -> bool:
        try:
            with self._connection() as conn:
                cur = conn.cursor()
                if self.use_postgres:
                    cur.execute("SELECT 1 FROM finagent.audit_events LIMIT 1")
                else:
                    cur.execute(
                        "CREATE TABLE IF NOT EXISTS audit_events (entry_index INTEGER PRIMARY KEY,event_id TEXT UNIQUE,payload TEXT NOT NULL)"
                    )
                conn.commit()
            self._db_available = True
            return True
        except Exception:
            self._db_available = False
            return False

    def close(self) -> None:
        return None

    def _load_tail(self) -> AuditEntry | None:
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT payload FROM {self.table} ORDER BY entry_index DESC LIMIT 1"
            )
            row = cur.fetchone()
        return (
            AuditEntry.model_validate(
                row[0] if isinstance(row[0], dict) else json.loads(row[0])
            )
            if row
            else None
        )

    def record_transition(
        self,
        trace_id: str,
        node_name: str,
        actor_id: str,
        action_type: str,
        state_payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> AuditEntry:
        if not self._db_available and not self.init_schema():
            raise RuntimeError(
                "Durable audit storage is unavailable; the transition was not committed."
            )
        marker = "%s" if self.use_postgres else "?"
        event_id = (metadata or {}).get("event_id")
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT pg_advisory_xact_lock(70621891)"
                if self.use_postgres
                else "BEGIN IMMEDIATE"
            )
            if event_id:
                cur.execute(
                    f"SELECT payload FROM {self.table} WHERE event_id={marker}",
                    (event_id,),
                )
                existing = cur.fetchone()
                if existing:
                    conn.commit()
                    return AuditEntry.model_validate(
                        existing[0]
                        if isinstance(existing[0], dict)
                        else json.loads(existing[0])
                    )
            cur.execute(
                f"SELECT payload FROM {self.table} ORDER BY entry_index DESC LIMIT 1"
            )
            row = cur.fetchone()
            local = MerkleAuditLedger()
            if row:
                local.entries.append(
                    AuditEntry.model_validate(
                        row[0] if isinstance(row[0], dict) else json.loads(row[0])
                    )
                )
            entry = local.record_transition(
                trace_id, node_name, actor_id, action_type, state_payload, metadata
            )
            payload = entry.model_dump()
            cur.execute(
                f"INSERT INTO {self.table}(entry_index,event_id,payload) VALUES ({marker},{marker},{marker})",
                (
                    entry.entry_index,
                    event_id,
                    Json(payload) if self.use_postgres else json.dumps(payload),
                ),
            )
            conn.commit()
        self.entries.append(entry)
        return entry

    def verify_integrity(self, from_db: bool = True) -> tuple[bool, str]:
        if not from_db:
            return super().verify_integrity()
        try:
            with self._connection() as conn:
                cur = conn.cursor()
                cur.execute(f"SELECT payload FROM {self.table} ORDER BY entry_index")
                rows = cur.fetchall()
            local = MerkleAuditLedger()
            local.entries = [
                AuditEntry.model_validate(
                    row[0] if isinstance(row[0], dict) else json.loads(row[0])
                )
                for row in rows
            ]
            return local.verify_integrity()
        except Exception:
            return False, "Durable audit verification is unavailable."
