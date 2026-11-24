from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

import redis
from langgraph.checkpoint.base import BaseCheckpointSaver

from agent.state.schema import AgentStateV1

_savers: dict[str, BaseCheckpointSaver] = {}
_lock = threading.RLock()


class WorkingMemoryManager:
    def __init__(
        self, redis_url: str | None = None, default_ttl_seconds: int = 86400
    ) -> None:
        self.redis_url: str = redis_url or str(
            os.getenv("REDIS_URL") or "redis://localhost:6379/0"
        )
        self.default_ttl = default_ttl_seconds
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = redis.Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
        return self._client

    def save_state(self, session_id: str, state: AgentStateV1) -> None:
        self._get_client().setex(
            f"finagent:working:{session_id}", self.default_ttl, state.model_dump_json()
        )

    def load_state(self, session_id: str) -> AgentStateV1 | None:
        raw = self._get_client().get(f"finagent:working:{session_id}")
        return AgentStateV1.model_validate(json.loads(raw)) if raw else None

    def clear_state(self, session_id: str) -> None:
        self._get_client().delete(f"finagent:working:{session_id}")


def get_checkpointer(redis_url: str | None = None) -> BaseCheckpointSaver:
    backend = str(
        os.getenv("FINAGENT_CHECKPOINT_BACKEND")
        or ("postgres" if os.getenv("DATABASE_URL") else "sqlite")
    )
    if backend == "postgres":
        target = str(os.getenv("DATABASE_URL") or "")
    elif backend == "redis":
        target = redis_url or str(os.getenv("REDIS_URL") or "redis://localhost:6379/0")
    else:
        target = str(
            os.getenv("FINAGENT_CHECKPOINT_PATH") or "data/runtime/checkpoints.sqlite3"
        )
    key = backend + ":" + target
    with _lock:
        saver: BaseCheckpointSaver
        if key in _savers:
            return _savers[key]
        if backend == "postgres":
            import psycopg
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg.rows import dict_row

            if not target:
                raise RuntimeError(
                    "DATABASE_URL is required for durable Postgres checkpoints."
                )
            conn = psycopg.connect(
                target,
                row_factory=dict_row,
                autocommit=True,
                prepare_threshold=0,
                connect_timeout=5,
                options="-c search_path=finagent,public -c statement_timeout=10000",
            )
            saver = PostgresSaver(conn)
            saver.setup()
        elif backend == "sqlite":
            from langgraph.checkpoint.sqlite import SqliteSaver

            Path(target).parent.mkdir(parents=True, exist_ok=True)
            sqlite_conn = sqlite3.connect(target, check_same_thread=False)
            sqlite_conn.execute("PRAGMA journal_mode=WAL")
            saver = SqliteSaver(sqlite_conn)
            saver.setup()
        elif backend == "redis":
            from langgraph.checkpoint.redis import RedisSaver

            client = redis.Redis.from_url(
                target, socket_connect_timeout=3, socket_timeout=3
            )
            client.ping()
            saver = RedisSaver(redis_client=client)
            saver.setup()
        else:
            raise RuntimeError(
                "Unsupported checkpoint backend. Durable sqlite, postgres or redis is required."
            )
        _savers[key] = saver
        return saver
