"""Token cost accounting and streaming latency profiler."""

from __future__ import annotations

import math
from typing import Any
from pydantic import BaseModel, Field

# Pricing per 1,000 tokens (USD)
MODEL_PRICING_TABLE: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "all-minilm-l6-v2": {"input": 0.0, "output": 0.0},  # Local embeddings
    "default": {"input": 0.002, "output": 0.008},
}


class TokenUsage(BaseModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)


class LatencyPercentiles(BaseModel):
    count: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


def calculate_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> TokenUsage:
    """Calculates dollar expenditure for an LLM invocation."""
    key = model_name.lower().strip()
    pricing = MODEL_PRICING_TABLE.get(key, MODEL_PRICING_TABLE["default"])

    input_cost = (prompt_tokens / 1000.0) * pricing["input"]
    output_cost = (completion_tokens / 1000.0) * pricing["output"]
    total_cost = round(input_cost + output_cost, 6)

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        estimated_cost_usd=total_cost,
    )


class LatencyProfiler:
    """Computes streaming quantiles and tail latencies over execution times."""

    def __init__(self) -> None:
        self.observations: list[float] = []

    def record(self, latency_ms: float) -> None:
        """Records an execution latency in milliseconds."""
        if latency_ms >= 0.0:
            self.observations.append(latency_ms)

    def compute_percentiles(self) -> LatencyPercentiles:
        """Computes sample quantiles across recorded observations."""
        n = len(self.observations)
        if n == 0:
            return LatencyPercentiles(
                count=0,
                mean_ms=0.0,
                p50_ms=0.0,
                p90_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                max_ms=0.0,
            )

        sorted_vals = sorted(self.observations)
        mean_val = sum(sorted_vals) / n

        def get_quantile(q: float) -> float:
            idx = int(math.ceil(q * n)) - 1
            return sorted_vals[max(0, min(idx, n - 1))]

        return LatencyPercentiles(
            count=n,
            mean_ms=round(mean_val, 2),
            p50_ms=round(get_quantile(0.50), 2),
            p90_ms=round(get_quantile(0.90), 2),
            p95_ms=round(get_quantile(0.95), 2),
            p99_ms=round(get_quantile(0.99), 2),
            max_ms=round(sorted_vals[-1], 2),
        )

    def reset(self) -> None:
        self.observations.clear()