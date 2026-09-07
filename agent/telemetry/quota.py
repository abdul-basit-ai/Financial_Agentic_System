"""LangSmith free-tier quota governor and sample rate controller.

Guards the 5,000 monthly free trace ceiling by:
1. Enforcing local trace budget thresholds (default 4,500 safety ceiling).
2. Applying deterministic or random sample rates for high-throughput batch evaluations.
3. Automatically switching projects between interactive development and batch evaluation.
"""

from __future__ import annotations

import os
import random
from typing import Any
from pydantic import BaseModel, Field

DEFAULT_MONTHLY_FREE_QUOTA = 5000
DEFAULT_SAFETY_CEILING = 4500  # Leaves 500 traces for ad-hoc debugging


class QuotaStatus(BaseModel):
    traces_recorded: int
    max_budget: int
    remaining_budget: int
    tracing_enabled: bool
    is_budget_exceeded: bool
    current_project: str


class LangSmithQuotaGuard:
    """Manages LangSmith trace dispatch budgets to prevent exceeding free tier limits."""

    def __init__(
        self,
        max_budget: int = DEFAULT_SAFETY_CEILING,
        sample_rate: float = 1.0,
        interactive_project: str = "finagent-interactive",
        eval_project: str = "finagent-eval",
    ) -> None:
        self.max_budget = max_budget
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self.interactive_project = interactive_project
        self.eval_project = eval_project
        self.trace_counter = 0

    def should_trace(self, is_eval_mode: bool = False) -> bool:
        """Determines whether an execution run should emit a LangSmith trace."""
        api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
        if not api_key:
            return False

        if self.trace_counter >= self.max_budget:
            return False

        # In evaluation mode, apply sampling to conserve quota
        if is_eval_mode:
            if self.sample_rate < 1.0 and random.random() > self.sample_rate:
                return False

        return True

    def record_trace(self, actually_traced: bool = True) -> None:
        """Increments the active trace counter.

        Only counts traces that were actually dispatched to LangSmith —
        counting locally-collected-only spans would inflate the recorded
        total and trip the budget ceiling early with real quota remaining.
        """
        if actually_traced:
            self.trace_counter += 1

    def configure_environment(self, is_eval_mode: bool = False) -> None:
        """Sets LangSmith environment variables to reflect current project and state."""
        tracing_active = self.should_trace(is_eval_mode=is_eval_mode)
        project = self.eval_project if is_eval_mode else self.interactive_project

        os.environ["LANGCHAIN_TRACING_V2"] = "true" if tracing_active else "false"
        os.environ["LANGSMITH_TRACING"] = "true" if tracing_active else "false"
        os.environ["LANGSMITH_PROJECT"] = project
        os.environ["LANGCHAIN_PROJECT"] = project

    def get_status(self, is_eval_mode: bool = False) -> QuotaStatus:
        project = self.eval_project if is_eval_mode else self.interactive_project
        return QuotaStatus(
            traces_recorded=self.trace_counter,
            max_budget=self.max_budget,
            remaining_budget=max(0, self.max_budget - self.trace_counter),
            tracing_enabled=self.should_trace(is_eval_mode=is_eval_mode),
            is_budget_exceeded=self.trace_counter >= self.max_budget,
            current_project=project,
        )

    def reset_counter(self) -> None:
        """Resets trace counter for testing or new billing cycles."""
        self.trace_counter = 0


# Global singleton instance
QUOTA_GUARD = LangSmithQuotaGuard()