from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import Json, execute_values


class AdmissionLimitError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _decode(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else json.loads(value)


class Transaction:
    def __init__(self, store: RecordStore, cursor: Any, owner: str):
        self.store, self.cursor, self.owner = store, cursor, owner

    def get(
        self, kind: str, record_id: str, *, lock: bool = False
    ) -> dict[str, Any] | None:
        p = self.store.marker
        suffix = " FOR UPDATE" if lock and self.store.database_url else ""
        self.cursor.execute(
            f"SELECT payload FROM {self.store.table} WHERE kind={p} AND id={p} AND owner_id={p}{suffix}",
            (kind, record_id, self.owner),
        )
        row = self.cursor.fetchone()
        return _decode(row[0]) if row else None

    def put(
        self,
        kind: str,
        record_id: str,
        payload: dict[str, Any],
        *,
        create_only: bool = False,
    ) -> bool:
        p = self.store.marker
        conflict = (
            "DO NOTHING"
            if create_only
            else "DO UPDATE SET payload=excluded.payload,updated_at=excluded.updated_at"
        )
        self.cursor.execute(
            f"INSERT INTO {self.store.table}(kind,id,owner_id,payload,updated_at) VALUES ({p},{p},{p},{p},{p}) ON CONFLICT(kind,id,owner_id) {conflict}",
            (kind, record_id, self.owner, self.store.encode(payload), utc_now()),
        )
        return bool(self.cursor.rowcount)

    def delete(self, kind: str, record_id: str) -> None:
        p = self.store.marker
        self.cursor.execute(
            f"DELETE FROM {self.store.table} WHERE kind={p} AND id={p} AND owner_id={p}",
            (kind, record_id, self.owner),
        )

    def evidence(self, document_ids: Sequence[str]) -> list[dict[str, Any]]:
        if not document_ids:
            return []
        p = self.store.marker
        field = self.store.json_text("document_id")
        slots = ",".join([p] * len(document_ids))
        self.cursor.execute(
            f"SELECT payload FROM {self.store.table} WHERE kind='evidence' AND owner_id={p} AND {field} IN ({slots}) ORDER BY id",
            (self.owner, *document_ids),
        )
        return [_decode(row[0]) for row in self.cursor.fetchall()]

    def clear_evidence(self, document_id: str) -> None:
        p = self.store.marker
        field = self.store.json_text("document_id")
        self.cursor.execute(
            f"DELETE FROM {self.store.table} WHERE kind='evidence' AND owner_id={p} AND {field}={p}",
            (self.owner, document_id),
        )

    def replace_evidence(self, document_id: str, rows: list[dict[str, Any]]) -> None:
        if any(r.get("document_id") != document_id for r in rows):
            raise ValueError("Report and evidence identifiers disagree.")
        self.clear_evidence(document_id)
        if not rows:
            return
        stamp = utc_now()
        values = [
            ("evidence", r["evidence_id"], self.owner, self.store.encode(r), stamp)
            for r in rows
        ]
        sql = f"INSERT INTO {self.store.table}(kind,id,owner_id,payload,updated_at) VALUES "
        if self.store.database_url:
            execute_values(self.cursor, sql + "%s", values, page_size=500)
        else:
            self.cursor.executemany(sql + "(?,?,?,?,?)", values)

    def count(self, kind: str, *, pending: bool = False) -> int:
        p = self.store.marker
        predicate = (
            f" AND {self.store.json_text('status')} IN ('QUEUED','RUNNING')"
            if pending
            else ""
        )
        self.cursor.execute(
            f"SELECT COUNT(*) FROM {self.store.table} WHERE kind={p} AND owner_id={p}{predicate}",
            (kind, self.owner),
        )
        return int(self.cursor.fetchone()[0])

    def find_document(self, digest: str) -> dict[str, Any] | None:
        p = self.store.marker
        self.cursor.execute(
            f"SELECT payload FROM {self.store.table} WHERE kind='documents' AND owner_id={p} AND {self.store.json_text('sha256')}={p} ORDER BY updated_at DESC LIMIT 1",
            (self.owner, digest),
        )
        row = self.cursor.fetchone()
        return _decode(row[0]) if row else None

    def admit(self) -> None:
        now = time.time()
        prior = self.get("quota", "submissions") or {}
        recent = [t for t in prior.get("timestamps", []) if t > now - 60]
        if len(recent) >= int(os.getenv("MAX_QUERIES_PER_MINUTE", "20")):
            raise AdmissionLimitError("Too many requests. Try again shortly.")
        if self.count("jobs", pending=True) >= int(os.getenv("MAX_PENDING_JOBS", "8")):
            raise AdmissionLimitError("Your pending work limit has been reached.")
        self.put("quota", "submissions", {"timestamps": [*recent, now]})

    def lease_current(self, job: dict[str, Any]) -> bool:
        record = self.get("jobs", job["job_id"], lock=True)
        return bool(
            record
            and record.get("status") == "RUNNING"
            and record.get("lease_token") == job.get("lease_token")
            and record.get("lease_until", "") > utc_now()
        )


class RecordStore:
    def __init__(self, database_url: str | None = None, path: str | None = None):
        self.database_url = database_url or os.getenv("DATABASE_URL", "")
        self.path = path or str(
            os.getenv("FINAGENT_DB_PATH") or "data/runtime/workspace.sqlite3"
        )
        if not self.database_url:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            with self.connection() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS workspace_records (kind TEXT NOT NULL,id TEXT NOT NULL,owner_id TEXT NOT NULL,payload TEXT NOT NULL,updated_at TEXT NOT NULL,PRIMARY KEY(kind,id,owner_id))"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS workspace_owner_kind ON workspace_records(owner_id,kind)"
                )
                conn.commit()

    @contextmanager
    def connection(self) -> Iterator[Any]:
        conn: Any
        if self.database_url:
            conn = psycopg2.connect(
                self.database_url,
                connect_timeout=5,
                options="-c statement_timeout=10000",
            )
        else:
            conn = sqlite3.connect(self.path, timeout=10)
        try:
            yield conn
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def transaction(self, owner: str, *, write: bool = True) -> Iterator[Transaction]:
        with self.connection() as conn:
            cursor = conn.cursor()
            if write and self.database_url:
                cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (owner,))
            elif not self.database_url:
                cursor.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield Transaction(self, cursor, owner)
            conn.commit()

    @property
    def table(self) -> str:
        return (
            "finagent.workspace_records" if self.database_url else "workspace_records"
        )

    @property
    def marker(self) -> str:
        return "%s" if self.database_url else "?"

    def json_text(self, key: str) -> str:
        if key not in {"status", "lease_until", "document_id", "sha256"}:
            raise ValueError("Unsupported record field")
        return (
            f"payload->>'{key}'"
            if self.database_url
            else f"json_extract(payload,'$.{key}')"
        )

    def encode(self, payload: dict[str, Any]) -> Any:
        def serialize(data: Any) -> str:
            return json.dumps(data, allow_nan=False, default=str)

        return (
            Json(payload, dumps=serialize) if self.database_url else serialize(payload)
        )

    def put(
        self,
        kind: str,
        record_id: str,
        owner_id: str,
        payload: dict[str, Any],
        create_only: bool = False,
    ) -> bool:
        with self.transaction(owner_id) as tx:
            return tx.put(kind, record_id, payload, create_only=create_only)

    def get(self, kind: str, record_id: str, owner_id: str) -> dict[str, Any] | None:
        with self.transaction(owner_id, write=False) as tx:
            return tx.get(kind, record_id)

    def list(
        self,
        kind: str,
        owner_id: str,
        limit: int | None = 500,
        *,
        paused_only: bool = False,
    ) -> list[dict[str, Any]]:
        p = self.marker
        conditions = ""
        if paused_only:
            conditions = (
                " AND payload->>'is_paused'='true'"
                if self.database_url
                else " AND json_extract(payload,'$.is_paused')=1"
            )
        limit_clause = f" LIMIT {p}" if limit is not None else ""
        parameters: tuple[Any, ...] = (
            (kind, owner_id, limit) if limit is not None else (kind, owner_id)
        )
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT payload FROM {self.table} WHERE kind={p} AND owner_id={p}{conditions} ORDER BY updated_at DESC{limit_clause}",
                parameters,
            )
            return [_decode(row[0]) for row in cur.fetchall()]

    def delete(self, kind: str, record_id: str, owner_id: str) -> None:
        with self.transaction(owner_id) as tx:
            tx.delete(kind, record_id)

    def claim_job(
        self, worker_id: str, lease_seconds: int = 300
    ) -> dict[str, Any] | None:
        with self.connection() as conn:
            cur = conn.cursor()
            if not self.database_url:
                cur.execute("BEGIN IMMEDIATE")
            p = self.marker
            suffix = " FOR UPDATE SKIP LOCKED" if self.database_url else ""
            status = self.json_text("status")
            lease = self.json_text("lease_until")
            now = datetime.now(UTC)
            cur.execute(
                f"SELECT id,owner_id,payload FROM {self.table} WHERE kind='jobs' AND ({status}='QUEUED' OR ({status}='RUNNING' AND COALESCE({lease},'')<{p})) ORDER BY updated_at LIMIT 1{suffix}",
                (now.isoformat(),),
            )
            row = cur.fetchone()
            if row is None:
                return None
            job = _decode(row[2])
            job.update(
                status="RUNNING",
                worker_id=worker_id,
                lease_token=uuid.uuid4().hex,
                lease_until=(now + timedelta(seconds=lease_seconds)).isoformat(),
                attempts=int(job.get("attempts", 0)) + 1,
                updated_at=now.isoformat(),
            )
            tx = Transaction(self, cur, row[1])
            tx.put("jobs", row[0], job)
            conn.commit()
            return {**job, "owner_id": row[1]}

    def renew(self, owner: str, job: dict[str, Any], seconds: int) -> bool:
        with self.transaction(owner) as tx:
            if not tx.lease_current(job):
                return False
            persisted = tx.get("jobs", job["job_id"])
            if not persisted:
                return False
            renewed = {
                **persisted,
                "lease_until": (
                    datetime.now(UTC) + timedelta(seconds=seconds)
                ).isoformat(),
            }
            tx.put("jobs", job["job_id"], renewed)
            return True

    def ping(self) -> bool:
        try:
            with self.connection() as conn:
                cur = conn.cursor()
                cur.execute(f"SELECT 1 FROM {self.table} LIMIT 1")
            return True
        except Exception:
            return False
