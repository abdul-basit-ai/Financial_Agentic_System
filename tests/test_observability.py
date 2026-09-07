"""Validation suite for Phase 12 Observability, LangSmith Quota Guards, and Telemetry.

Verifies:
1. LangSmith quota ceilings and budget preservation thresholds.
2. Sample-rate throttling during batch evaluations.
3. Accurate token cost accounting across model pricing tiers.
4. Latency quantile estimations (P50, P90, P95, P99).
5. Non-intrusive tracing decorators and local fallback trace collection.
"""

from __future__ import annotations

import os
import pytest

from agent.telemetry import (
    LOCAL_TRACER,
    QUOTA_GUARD,
    LatencyProfiler,
    calculate_cost,
    init_telemetry,
    trace_operation,
)


# =====================================================================
# 1. LangSmith Free-Tier Quota & Budget Tests
# =====================================================================


def test_quota_guard_budget_ceiling_enforcement() -> None:
    QUOTA_GUARD.reset_counter()
    QUOTA_GUARD.max_budget = 5  # Set small budget for test

    # Simulate API Key presence
    os.environ["LANGSMITH_API_KEY"] = "lsv2_test_token"

    for _ in range(5):
        assert QUOTA_GUARD.should_trace() is True
        QUOTA_GUARD.record_trace()

    # Once 5 is reached, tracing must shut off to protect free quota
    assert QUOTA_GUARD.should_trace() is False
    status = QUOTA_GUARD.get_status()
    assert status.is_budget_exceeded is True
    assert status.remaining_budget == 0


def test_quota_guard_evaluation_project_isolation() -> None:
    QUOTA_GUARD.reset_counter()
    QUOTA_GUARD.max_budget = 4500
    os.environ["LANGSMITH_API_KEY"] = "lsv2_test_token"

    # Interactive mode configuration
    init_telemetry(is_eval_mode=False)
    assert os.getenv("LANGSMITH_PROJECT") == "finagent-interactive"

    # Batch evaluation mode configuration (switches project)
    init_telemetry(is_eval_mode=True)
    assert os.getenv("LANGSMITH_PROJECT") == "finagent-eval"


def test_quota_guard_sample_rate_governor() -> None:
    QUOTA_GUARD.reset_counter()
    QUOTA_GUARD.max_budget = 1000
    QUOTA_GUARD.sample_rate = 0.0  # 0% sample rate
    os.environ["LANGSMITH_API_KEY"] = "lsv2_test_token"

    # In eval mode with sample_rate=0.0, should_trace must return False
    assert QUOTA_GUARD.should_trace(is_eval_mode=True) is False


# =====================================================================
# 2. Token Economics & Cost Calculation Tests
# =====================================================================


def test_calculate_cost_gpt_4o() -> None:
    # GPT-4o: $0.005/1k input, $0.015/1k output
    usage = calculate_cost(
        model_name="gpt-4o",
        prompt_tokens=2000,
        completion_tokens=1000,
    )
    assert usage.prompt_tokens == 2000
    assert usage.completion_tokens == 1000
    assert usage.total_tokens == 3000
    # Expected: (2 * 0.005) + (1 * 0.015) = 0.01 + 0.015 = $0.025
    assert usage.estimated_cost_usd == 0.025


def test_calculate_cost_claude_haiku() -> None:
    # Claude-3-Haiku: $0.00025/1k input, $0.00125/1k output
    usage = calculate_cost(
        model_name="claude-3-haiku",
        prompt_tokens=4000,
        completion_tokens=2000,
    )
    # Expected: (4 * 0.00025) + (2 * 0.00125) = 0.001 + 0.0025 = $0.0035
    assert usage.estimated_cost_usd == 0.0035


# =====================================================================
# 3. Latency Percentiles Profiler Tests
# =====================================================================


def test_latency_profiler_quantiles() -> None:
    profiler = LatencyProfiler()

    # Populate 100 observations from 1ms to 100ms
    for ms in range(1, 101):
        profiler.record(float(ms))

    stats = profiler.compute_percentiles()
    assert stats.count == 100
    assert stats.mean_ms == 50.5
    assert stats.p50_ms == 50.0
    assert stats.p90_ms == 90.0
    assert stats.p95_ms == 95.0
    assert stats.p99_ms == 99.0
    assert stats.max_ms == 100.0


# =====================================================================
# 4. Tracing Decorator & Span Fallback Tests
# =====================================================================


def test_trace_operation_decorator() -> None:
    LOCAL_TRACER.clear()
    QUOTA_GUARD.reset_counter()

    @trace_operation(name="test_math_op", run_type="tool", metadata={"metric": "margin"})
    def add_numbers(a: int, b: int) -> int:
        return a + b

    res = add_numbers(10, 20)
    assert res == 30

    # Verify span was recorded locally
    assert len(LOCAL_TRACER.spans) == 1
    span = LOCAL_TRACER.spans[0]
    assert span.name == "test_math_op"
    assert span.run_type == "tool"
    assert span.outputs == "30"
    assert span.metadata["metric"] == "margin"
    assert span.duration_ms >= 0.0