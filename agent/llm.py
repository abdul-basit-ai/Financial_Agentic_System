"""LLM provider with graceful degradation.

Single access point for every LLM call in the agent (Phase 6 planner, Phase 14
synthesizer). Design contract:

- Lazy model construction from environment: OPENAI_API_KEY or
  OPENROUTER_API_KEY, OPENAI_MODEL (default gpt-4o-mini), OPENAI_BASE_URL
  (auto-set to the OpenRouter endpoint when only an OpenRouter key is present,
  so Azure/OpenRouter/local gateways need no extra config).
- UNAVAILABLE is not an error: with no usable key, callers get None and fall
  back to the deterministic planner/synthesizer paths, so evaluation and CI
  stay runnable offline.
- FAILURES are loud but non-fatal: invocation errors print a warning and
  return None — the agent degrades to rules rather than crashing a run.
- Every successful call returns token counts and estimated cost so nodes can
  log per-query spend (Phase 12 dashboard hook).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from agent.telemetry.metrics import calculate_cost

_PLACEHOLDER_KEYS = {"", "sk-...", "changeme", "your-api-key"}


@dataclass
class LLMCallResult:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _api_key() -> str:
    """Configured provider key: OpenAI or OpenRouter (whichever is set)."""
    _load_env_file()
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        value = os.getenv(name, "").strip()
        if value.lower() not in _PLACEHOLDER_KEYS:
            return value
    return ""


def is_llm_enabled() -> bool:
    """True when a usable OpenAI/OpenRouter API key is configured."""
    return bool(_api_key())


def _default_base_url() -> str:
    """OpenRouter endpoint when only an OpenRouter key is present; otherwise
    the OpenAI default (empty string = SDK default)."""
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key.lower() not in _PLACEHOLDER_KEYS:
        return ""
    if os.getenv("OPENROUTER_API_KEY", "").strip():
        return OPENROUTER_BASE_URL
    return ""


def _load_env_file() -> None:
    """Best-effort .env loading so local runs pick up OPENAI_API_KEY."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass


@dataclass
class _ModelHandle:
    model: Any
    model_name: str


_model_cache: _ModelHandle | None = None


def _get_model() -> _ModelHandle | None:
    """Constructs the chat model once per process; None when unconfigured."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache if _model_cache.model is not None else None

    _load_env_file()
    if not is_llm_enabled():
        return None

    try:
        from langchain_openai import ChatOpenAI

        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        kwargs: dict[str, Any] = {
            "model": model_name,
            "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0")),
            "timeout": float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30")),
            "max_retries": 1,
            "api_key": _api_key(),
        }
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or _default_base_url()
        if base_url:
            kwargs["base_url"] = base_url
        _model_cache = _ModelHandle(model=ChatOpenAI(**kwargs), model_name=model_name)
        return _model_cache
    except Exception as exc:
        print(
            f"[llm] Chat model unavailable ({type(exc).__name__}: {exc}); "
            f"falling back to deterministic behavior.",
            flush=True,
        )
        return None


_NUMERAL_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_numerals(text: str) -> set[str]:
    """Numeric literals in text, comma separators stripped ("1,200" -> "1200")."""
    return {m.replace(",", "") for m in _NUMERAL_RE.findall(str(text))}


def invoke_llm(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
) -> LLMCallResult | None:
    """Runs one completion; returns None when LLM is unconfigured or fails.

    Never raises — callers treat None as "use the deterministic fallback".
    """
    handle = _get_model()
    if handle is None:
        return None

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = handle.model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ],
            max_tokens=max_tokens,
        )
    except Exception as exc:
        print(
            f"[llm] invocation failed ({type(exc).__name__}: {exc}); "
            f"falling back to deterministic behavior.",
            flush=True,
        )
        return None

    content = str(getattr(response, "content", "") or "")
    usage = getattr(response, "usage_metadata", None) or {}
    prompt_tokens = int(usage.get("input_tokens", 0) or 0)
    completion_tokens = int(usage.get("output_tokens", 0) or 0)
    cost = calculate_cost(handle.model_name, prompt_tokens, completion_tokens)

    return LLMCallResult(
        content=content,
        model=handle.model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost.estimated_cost_usd,
    )
