"""Unit detection and normalization for financial numeric values."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

UNIT_MULTIPLIERS = {
    "thousand": 1_000.0,
    "thousands": 1_000.0,
    "million": 1_000_000.0,
    "millions": 1_000_000.0,
    "billion": 1_000_000_000.0,
    "billions": 1_000_000_000.0,
    "trillion": 1_000_000_000_000.0,
    "trillions": 1_000_000_000_000.0,
}

UNIT_RE = re.compile(r"\b(thousand|thousands|million|millions|billion|billions|trillion|trillions)\b", re.IGNORECASE)


def detect_unit_label(*texts: Any) -> str:
    for t in texts:
        s = str(t or "")
        m = UNIT_RE.search(s)
        if m:
            return m.group(1).lower()
    return "base"


def normalize_value(value: Optional[float], unit_label: str) -> Optional[float]:
    if value is None:
        return None
    mult = UNIT_MULTIPLIERS.get((unit_label or "").lower(), 1.0)
    return value * mult


def normalize_with_context(value: Optional[float], *context_texts: Any) -> Dict[str, Optional[float]]:
    unit = detect_unit_label(*context_texts)
    return {
        "unit_label": unit,
        "value_base": normalize_value(value, unit),
    }
