"""Small train-only FinQA program exemplar library for prompt conditioning."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

_WORD_RE = re.compile(r"[a-z]{3,}")
_DEFAULT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "program_exemplars.json"
)


def operation_family(program: str) -> str:
    match = re.search(r"\b(table_average|add|subtract|multiply|divide)\s*\(", program)
    return match.group(1) if match else "other"


@lru_cache(maxsize=1)
def load_exemplars(path: str | None = None) -> tuple[dict[str, Any], ...]:
    source = Path(path) if path else _DEFAULT_PATH
    if not source.exists():
        return ()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    return tuple(item for item in payload if isinstance(item, dict))


def format_exemplars(question: str, limit: int = 4) -> str:
    """Selects train examples by operation family and lexical question overlap."""
    examples = load_exemplars()
    if not examples:
        return "(no train exemplars installed)"
    words = set(_WORD_RE.findall(question.casefold()))
    ranked = sorted(
        examples,
        key=lambda item: (
            len(
                words & set(_WORD_RE.findall(str(item.get("question", "")).casefold()))
            ),
            -len(str(item.get("question", ""))),
        ),
        reverse=True,
    )
    lines = [
        f"Example {idx}: {item.get('question')} -> {item.get('program')}"
        for idx, item in enumerate(ranked[:limit], start=1)
    ]
    return "\n".join(lines)
