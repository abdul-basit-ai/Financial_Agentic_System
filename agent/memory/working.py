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

    Prefers RedisSaver so paused/interrupted state survives process restarts
    (required by Phase 8 HITL). Falls back to MemorySaver only when the Redis
    package is missing or the server is unreachable — never silently: the
    fallback reason is printed for Phase 12 tracing.

    Note: the RedisSaver instance returned here has had .setup() called, which
    creates its search indices. Requires a Redis server with the RediSearch +
    ReJSON modules (redis-stack image), not plain redis-server.
    """
    target_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        from langgraph.checkpoint.redis import RedisSaver

        client = redis.Redis.from_url(target_url)
        client.ping()
        saver = RedisSaver(redis_client=client)
        saver.setup()
        return saver
    except ImportError as exc:
        print(
            f"[working_memory] RedisSaver unavailable ({exc}); falling back to "
            f"MemorySaver. Paused state will NOT survive process restarts.",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[working_memory] Redis unreachable at {target_url} ({exc}); "
            f"falling back to MemorySaver. Paused state will NOT survive "
            f"process restarts.",
            flush=True,
        )
    return MemorySaver()