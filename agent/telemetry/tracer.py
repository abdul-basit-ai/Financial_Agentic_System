"""Unified tracer integrating LangSmith with local fallback telemetry."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar
from pydantic import BaseModel, Field

from agent.telemetry.quota import QUOTA_GUARD

T = TypeVar("T")

# Safe import of LangSmith traceable
try:
    from langsmith import traceable as ls_traceable
except ImportError:
    ls_traceable = None


class TraceSpanRecord(BaseModel):
    name: str
    run_type: str
    inputs: dict[str, Any]
    outputs: Any | None = None
    error: str | None = None
    start_time: float
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class LocalTraceCollector:
    """In-memory telemetry collector used when LangSmith is disabled or quota is capped."""

    def __init__(self) -> None:
        self.spans: list[TraceSpanRecord] = []

    def add_span(self, span: TraceSpanRecord) -> None:
        self.spans.append(span)

    def clear(self) -> None:
        self.spans.clear()


LOCAL_TRACER = LocalTraceCollector()


def init_telemetry(
    is_eval_mode: bool = False,
    sample_rate: float = 1.0,
    max_budget: int = 4500,
) -> None:
    """Bootstraps telemetry environment variables and quota configurations."""
    QUOTA_GUARD.max_budget = max_budget
    QUOTA_GUARD.sample_rate = sample_rate
    QUOTA_GUARD.configure_environment(is_eval_mode=is_eval_mode)


def trace_operation(
    name: str,
    run_type: str = "tool",
    metadata: dict[str, Any] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Universal decorator tracing functions to LangSmith or the local collector.

    The LangSmith wrapping decision is made at CALL time (inside the wrapper),
    not decoration time, so environment configuration (API key, tracing flags)
    set after module import — e.g. by init_telemetry — takes effect correctly.
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.perf_counter()
            err_msg: str | None = None
            output: Any = None

            # Decide tracing at call time: env config may change after import
            should_send = QUOTA_GUARD.should_trace()
            if should_send:
                QUOTA_GUARD.record_trace(actually_traced=True)

            try:
                if should_send and ls_traceable is not None:
                    wrapped = ls_traceable(
                        name=name,
                        run_type=run_type,
                        metadata=metadata or {},
                    )(fn)
                    output = wrapped(*args, **kwargs)
                else:
                    output = fn(*args, **kwargs)
                return output
            except Exception as exc:
                err_msg = str(exc)
                raise
            finally:
                duration = (time.perf_counter() - start) * 1000.0
                LOCAL_TRACER.add_span(
                    TraceSpanRecord(
                        name=name,
                        run_type=run_type,
                        inputs={"args": str(args)[:200], "kwargs": str(kwargs)[:200]},
                        outputs=str(output)[:200] if output else None,
                        error=err_msg,
                        start_time=start,
                        duration_ms=round(duration, 3),
                        metadata={**(metadata or {}), "langsmith_sent": should_send},
                    )
                )

        return wrapper

    return decorator