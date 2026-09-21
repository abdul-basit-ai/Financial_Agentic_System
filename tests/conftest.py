"""Shared pytest configuration.

Forces deterministic (no-LLM) agent behavior during tests: .env may carry a
real OpenRouter/OpenAI key on dev machines, and without this pin the e2e graph
tests would issue real paid API calls and slow down/flake. Set
FINAGENT_TESTS_LLM=1 to opt into live-LLM test runs deliberately.
"""

from __future__ import annotations

import os


def pytest_configure(config) -> None:
    if os.getenv("FINAGENT_TESTS_LLM", "").strip() != "1":
        os.environ.setdefault("OPENAI_API_KEY", "")
        os.environ.setdefault("OPENROUTER_API_KEY", "")
