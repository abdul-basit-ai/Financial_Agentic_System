"""Entity extraction for FinQA records."""

from __future__ import annotations

import re
from typing import Any

YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
UNIT_RE = re.compile(
    r"\b(thousand|thousands|million|millions|billion|billions|trillion|trillions|percent|percentage|%)\b",
    re.IGNORECASE,
)

COMPANY_SUFFIXES = {
    "inc", "inc.", "corp", "corp.", "corporation", "co", "co.",
    "company", "ltd", "ltd.", "plc", "llc", "group", "holdings"
}

BLACKLIST_IDENTIFIERS = {
    "PAGE", "DOC", "DOCUMENT", "TABLE", "TEST", "TRAIN", "DEV",
    "FINQA", "FILE", "ITEM", "SECTION", "ANNUAL", "REPORT"
}


def _text_join(parts: list[Any]) -> str:
    out = []
    for p in parts:
        if isinstance(p, list):
            out.extend(str(x) for x in p)
        else:
            out.append(str(p))
    return " ".join(out)


def _extract_company_candidates(text: str) -> set[str]:
    entities: set[str] = set()
    tokens = text.split()
    for i in range(len(tokens) - 1):
        tok = tokens[i].strip(",.;:()[]{}")
        nxt = tokens[i + 1].strip(",.;:()[]{}")
        if tok and nxt and nxt.lower() in COMPANY_SUFFIXES and tok[0].isupper():
            entities.add(f"{tok} {nxt}")
    return entities


def _extract_company_identifier(filename: str) -> str:
    if not filename:
        return ""
    identifier = re.split(r"[/\\_]", str(filename), maxsplit=1)[0].strip()
    clean_id = identifier.upper()
    if clean_id in BLACKLIST_IDENTIFIERS or len(clean_id) < 2:
        return ""
    return clean_id


def extract_entities(
    question: str,
    filename: str,
    pre_text: list[str],
    post_text: list[str],
    table: list[list[Any]],
) -> dict[str, list[str]]:
    table_text = _text_join(table if isinstance(table, list) else [])
    all_text = _text_join([filename, question, pre_text, post_text, table_text])

    fiscal_years = sorted(set(YEAR_RE.findall(all_text)))
    units = sorted(set(u.lower() for u in UNIT_RE.findall(all_text)))

    metrics: set[str] = set()
    if isinstance(table, list):
        for row in table[1:]:
            if isinstance(row, list) and row and isinstance(row[0], str):
                metric = " ".join(row[0].split())
                if metric:
                    metrics.add(metric)

    companies = _extract_company_candidates(all_text)
    filename_company = _extract_company_identifier(filename)
    if filename_company:
        companies.add(filename_company)
    elif pre_text:
        candidates = re.findall(r"\b[A-Z]{2,5}\b", " ".join(pre_text))
        valid_candidates = [c for c in candidates if c not in BLACKLIST_IDENTIFIERS]
        companies.update(valid_candidates)

    return {
        "company_names": sorted(companies),
        "fiscal_years": fiscal_years,
        "metric_names": sorted(metrics),
        "units": units,
    }