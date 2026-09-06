"""Persistent tool metrics logging (Phase 12 cost dashboard / Phase 19 MLOps feed).

Appends one JSON line per instrumented tool call to data/metrics/tool_metrics.jsonl.
Format mirrors ToolResult plus trace correlation fields:
    {"ts_utc", "trace_id", "tool_name", "success", "latency_ms",
     "tokens_used", "cost_usd", "error"}

Thread-safe via a process-wide lock; never raises into the caller — a metrics
failure must not break a working tool call (fail-open by design).
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from typing import Any

from agent.tools.base import ToolResult

METRICS_DIR = os.path.join("data", "metrics")
METRICS_PATH = os.path.join(METRICS_DIR, "tool_metrics.jsonl")
_LOCK = threading.Lock()


def log_tool_result(
    result: ToolResult[Any],
    trace_id: str | None = None,
    path: str = METRICS_PATH,
) -> bool:
    """Appends a ToolResult to the JSONL metrics log. Returns success flag."""
    try:
        record = {
            "ts_utc": datetime.now(UTC).isoformat(),
            "trace_id": trace_id,
            "tool_name": result.tool_name,
            "success": result.success,
            "latency_ms": result.metrics.latency_ms,
            "tokens_used": result.metrics.tokens_used,
            "cost_usd": result.metrics.cost_usd,
            "error": result.error,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except Exception:
        # Metrics logging must never break tool execution (fail-open).
        return False


def read_metrics(path: str = METRICS_PATH) -> list[dict[str, Any]]:
    """Reads all recorded metrics (used by Phase 12 aggregation)."""
    if not os.path.exists(path):
        return []
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def summarize_metrics(path: str = METRICS_PATH) -> dict[str, Any]:
    """Per-tool aggregates: call count, success rate, total latency/cost."""
    metrics = read_metrics(path)
    by_tool: dict[str, dict[str, Any]] = {}
    for m in metrics:
        agg = by_tool.setdefault(
            m["tool_name"],
            {"calls": 0, "successes": 0, "total_latency_ms": 0.0, "total_cost_usd": 0.0},
        )
        agg["calls"] += 1
        agg["successes"] += int(bool(m["success"]))
        agg["total_latency_ms"] += m.get("latency_ms", 0.0)
        agg["total_cost_usd"] += m.get("cost_usd", 0.0)
    for agg in by_tool.values():
        agg["success_rate"] = agg["successes"] / agg["calls"] if agg["calls"] else 0.0
        del agg["successes"]
    return by_tool
