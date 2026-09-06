"""Per-tool concurrency limiting for parallel sub-task execution.

The plan (Phase 7) requires a hard concurrency bound so parallel Send branches
cannot blow through Neo4j/pgvector connection pools or API rate limits. This
module provides a process-wide semaphore registry keyed by tool name.

Design:
- Each tool gets a BoundedSemaphore (default: 4 concurrent executions per tool).
- Workers acquire BEFORE invoking the tool and release in a finally block.
- A non-blocking fast-path is tried first; if saturated, the worker blocks with
  a timeout and logs contention so Phase 12 dashboards can observe queuing.
- Wait time is recorded on the envelope metrics (contention visibility).

Phase 12 integration point: expose TOOL_CONCURRENCY_LIMITS + semaphore stats
via the metrics logger. Phase 19 can tune limits per environment.
"""

from __future__ import annotations

import threading
import time
from typing import Any

# Hard per-tool concurrency caps. Neo4j community default pool is 100; pgvector
# compose instance comfortably handles 10; safe_math is CPU-bound and cheap.
TOOL_CONCURRENCY_LIMITS: dict[str, int] = {
    "graph_retrieval": 4,
    "vector_retrieval": 4,
    "safe_math": 8,
}
DEFAULT_LIMIT = 4

# Max seconds a worker may wait for a slot before executing anyway with a
# contention warning. Prevents indefinite stalls if a downstream tool hangs.
MAX_WAIT_SECONDS = 30.0

_semaphores: dict[str, threading.BoundedSemaphore] = {}
_registry_lock = threading.Lock()
_contention_stats: dict[str, dict[str, float]] = {}


def _get_semaphore(tool_name: str) -> threading.BoundedSemaphore:
    with _registry_lock:
        if tool_name not in _semaphores:
            limit = TOOL_CONCURRENCY_LIMITS.get(tool_name, DEFAULT_LIMIT)
            _semaphores[tool_name] = threading.BoundedSemaphore(limit)
        return _semaphores[tool_name]


def acquire_tool_slot(tool_name: str) -> tuple[bool, float, str]:
    """Acquires a concurrency slot for the tool.

    Returns (acquired_immediately, wait_seconds, status) where status is one of:
      'immediate' - slot was free
      'waited'    - blocked, then acquired within MAX_WAIT_SECONDS
      'timeout'   - could not acquire in time (caller decides policy)
    """
    sem = _get_semaphore(tool_name)
    acquired = sem.acquire(blocking=False)
    if acquired:
        return True, 0.0, "immediate"

    start = time.perf_counter()
    acquired = sem.acquire(blocking=True, timeout=MAX_WAIT_SECONDS)
    waited = time.perf_counter() - start

    if acquired:
        _record_contention(tool_name, waited)
        return True, waited, "waited"
    _record_contention(tool_name, waited, timeout=True)
    return False, waited, "timeout"


def release_tool_slot(tool_name: str) -> None:
    _get_semaphore(tool_name).release()


def _record_contention(tool_name: str, waited: float, timeout: bool = False) -> None:
    with _registry_lock:
        stats = _contention_stats.setdefault(
            tool_name, {"total_waits": 0.0, "wait_events": 0.0, "timeouts": 0.0}
        )
        stats["total_waits"] += waited
        stats["wait_events"] += 1.0
        if timeout:
            stats["timeouts"] += 1.0


def get_contention_stats() -> dict[str, dict[str, float]]:
    """Read-only view for Phase 12 cost/latency dashboards."""
    with _registry_lock:
        return {
            tool: {
                "total_wait_seconds": round(s["total_waits"], 4),
                "wait_events": int(s["wait_events"]),
                "timeouts": int(s["timeouts"]),
            }
            for tool, s in _contention_stats.items()
        }


class ToolSlot:
    """Context manager acquiring a per-tool concurrency slot.

    Usage:
        with ToolSlot("graph_retrieval") as slot:
            result = graph_retrieval_tool(payload)
        # slot.wait_seconds / slot.status feed metrics logging
    """

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.wait_seconds: float = 0.0
        self.status: str = "not_acquired"
        self._acquired = False

    def __enter__(self) -> "ToolSlot":
        acquired, waited, status = acquire_tool_slot(self.tool_name)
        self._acquired = acquired
        self.wait_seconds = waited
        self.status = status
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._acquired:
            release_tool_slot(self.tool_name)
            self._acquired = False
