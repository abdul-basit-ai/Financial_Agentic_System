"""Working memory controller with Redis serialization and LangGraph checkpointing."""

from __future__ import annotations

import json
import os
from typing import Any
import redis
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from agent.state.schema import AgentStateV1


class WorkingMemoryManager:
    """Manages ephemeral working memory cache with TTL in Redis."""

    def __init__(
        self,
        redis_url: str | None = None,
        default_ttl_seconds: int = 86400,  # 24 hours
    ) -> None:
        self.redis_url = redis_url or os.getenv(
            "REDIS_URL", "redis://localhost:6379/0"
        )
        self.default_ttl = default_ttl_seconds
        self._client: redis.Redis | None = None

    def _get_client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.Redis.from_url(
                self.redis_url, decode_responses=True
            )
        return self._client

    def save_state(self, session_id: str, state: AgentStateV1) -> None:
        client = self._get_client()
        key = f"finagent:working:{session_id}"
        payload = state.model_dump_json()
        client.setex(key, self.default_ttl, payload)

    def load_state(self, session_id: str) -> AgentStateV1 | None:
        client = self._get_client()
        key = f"finagent:working:{session_id}"
        raw = client.get(key)
        if not raw:
            return None
        data = json.loads(str(raw))
        return AgentStateV1.model_validate(data)

    def clear_state(self, session_id: str) -> None:
        client = self._get_client()
        client.delete(f"finagent:working:{session_id}")


def get_checkpointer(redis_url: str | None = None) -> BaseCheckpointSaver:
    """Initializes checkpointer for LangGraph StateGraph persistence.

    Falls back to MemorySaver for local non-Redis environments.
    """
    try:
        from langgraph.checkpoint.redis import RedisSaver

        target_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        client = redis.Redis.from_url(target_url)
        client.ping()
        return RedisSaver(conn=client)
    except (ImportError, Exception):
        return MemorySaver()