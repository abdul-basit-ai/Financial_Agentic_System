"""Entity extraction for FinQA records."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set

YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
UNIT_RE = re.compile(r"\b(thousand|thousands|million|millions|billion|billions|trillion|trillions|percent|percentage|%)\b", re.IGNORECASE)

COMPANY_SUFFIXES = [
    "inc",
    "inc.",
    "corp",
    "corp.",
    "corporation",
    "co",
    "co.",
    "company",
    "ltd",
    "ltd.",
    "plc",
    "llc",
    "group",
    "holdings",
]

METRIC_PATTERNS = {
    "revenue": ["revenue", "sales", "net sales"],
    "net_income": ["net income", "net earnings", "net income (loss)", "income (loss)"],
    "operating_income": ["operating income", "income from operations"],
    "cash_flow": ["cash flow", "operating cash flow", "free cash flow"],
    "ebitda": ["ebitda"],
    "assets": ["assets", "total assets"],
    "liabilities": ["liabilities", "total liabilities"],
    "equity": ["equity", "shareholders' equity", "stockholders' equity"],
    "eps": ["eps", "earnings per share", "diluted eps", "basic eps"],
    "dividends": ["dividend", "dividends"],
    "shares": ["shares", "share count", "outstanding shares"],
    "interest_expense": ["interest expense"],
    "tax": ["tax", "income tax", "effective tax rate"],
}


def _text_join(parts: List[Any]) -> str:
    out = []
    for p in parts:
        if isinstance(p, list):
            out.extend(str(x) for x in p)
        else:
            out.append(str(p))
    return " ".join(out)


def _extract_company_candidates(text: str) -> Set[str]:
    entities: Set[str] = set()
    tokens = text.split()
    for i in range(len(tokens) - 1):
        tok = tokens[i].strip(",.;:()[]{}")
        nxt = tokens[i + 1].strip(",.;:()[]{}")
        if tok and nxt and nxt.lower() in COMPANY_SUFFIXES and tok[0].isupper():
            entities.add(f"{tok} {nxt}")
    return entities


def extract_entities(
    question: str,
    filename: str,
    pre_text: List[str],
    post_text: List[str],
    table: List[List[Any]],
) -> Dict[str, List[str]]:
    table_text = _text_join(table if isinstance(table, list) else [])
    all_text = _text_join([filename, question, pre_text, post_text, table_text])
    lower = all_text.lower()

    fiscal_years = sorted(set(YEAR_RE.findall(all_text)))
    units = sorted(set(u.lower() for u in UNIT_RE.findall(all_text)))

    metrics: Set[str] = set()
    for canonical, variants in METRIC_PATTERNS.items():
        if any(v in lower for v in variants):
            metrics.add(canonical)

    companies = sorted(_extract_company_candidates(all_text))

    return {
        "company_names": companies,
        "fiscal_years": fiscal_years,
        "metric_names": sorted(metrics),
        "units": units,
    }
