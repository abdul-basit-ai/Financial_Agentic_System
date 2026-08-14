"""Entity resolution helpers for financial metric canonicalization."""

from __future__ import annotations

from typing import Dict, Iterable, List

METRIC_SYNONYMS = {
    "net_income": [
        "net income",
        "net income (loss)",
        "income (loss)",
        "net earnings",
    ],
    "revenue": [
        "revenue",
        "net sales",
        "sales",
    ],
    "operating_income": [
        "operating income",
        "income from operations",
    ],
    "cash_flow": [
        "cash flow",
        "operating cash flow",
        "free cash flow",
    ],
    "eps": [
        "earnings per share",
        "eps",
        "diluted eps",
        "basic eps",
    ],
    "equity": [
        "shareholders' equity",
        "stockholders' equity",
        "equity",
    ],
}


def build_inverse_map() -> Dict[str, str]:
    inverse: Dict[str, str] = {}
    for canonical, variants in METRIC_SYNONYMS.items():
        inverse[canonical.lower()] = canonical
        for v in variants:
            inverse[v.lower()] = canonical
    return inverse


INVERSE_METRIC_MAP = build_inverse_map()


def resolve_metric_name(name: str) -> str:
    key = (name or "").strip().lower()
    if not key:
        return ""
    return INVERSE_METRIC_MAP.get(key, key.replace(" ", "_"))


def resolve_metric_names(names: Iterable[str]) -> List[str]:
    out = []
    seen = set()
    for n in names:
        c = resolve_metric_name(str(n))
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out
