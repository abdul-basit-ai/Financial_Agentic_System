"""Base schemas and metrics tracking for agent tools."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Generic, TypeVar
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ToolMetrics(BaseModel):
    """Execution telemetry for agent tools."""
    latency_ms: float = Field(..., description="Wall-clock latency in milliseconds")
    tokens_used: int = Field(default=0, description="Estimated or actual tokens consumed")
    cost_usd: float = Field(default=0.0, description="Estimated API cost in USD")


class ToolResult(BaseModel, Generic[T]):
    """Standardized envelope for tool execution results."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_name: str = Field(..., description="Identifier of the executing tool")
    success: bool = Field(..., description="Whether the tool execution succeeded without error")
    data: T | None = Field(default=None, description="Typed payload returned by the tool")
    error: str | None = Field(default=None, description="Error message if execution failed")
    metrics: ToolMetrics = Field(..., description="Telemetry and execution metrics")

    @classmethod
    def execute_instrumented(
        cls,
        tool_name: str,
        fn: Callable[..., T],
        *args: Any,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
        **kwargs: Any,
    ) -> ToolResult[T]:
        """Runs a callable, measures latency, and catches exceptions cleanly."""
        start_time = time.perf_counter()
        try:
            result_data = fn(*args, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return cls(
                tool_name=tool_name,
                success=True,
                data=result_data,
                error=None,
                metrics=ToolMetrics(
                    latency_ms=round(duration_ms, 3),
                    tokens_used=tokens_used,
                    cost_usd=cost_usd,
                ),
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return cls(
                tool_name=tool_name,
                success=False,
                data=None,
                error=f"{type(exc).__name__}: {str(exc)}",
                metrics=ToolMetrics(
                    latency_ms=round(duration_ms, 3),
                    tokens_used=tokens_used,
                    cost_usd=cost_usd,
                ),
            )