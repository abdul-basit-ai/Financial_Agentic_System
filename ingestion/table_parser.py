"""Table parsing helpers for multi-header, merged-like, and nested row labels."""

from __future__ import annotations

import re
from typing import Any

from ingestion.unit_normalizer import parse_amount, scale_hints

_PERIOD_HEADER = re.compile(
    r"\b(months?\s+ended|quarter|as\s+of|years?\s+ended)\b", re.I
)
_YEAR = re.compile(r"^(?:FY\s*)?(?:19|20)\d{2}$", re.I)
_LABEL_HEADER = re.compile(
    r"^(?:year|period|fiscal year|metric|line item|description)s?$", re.I
)


def _clean(x: Any) -> str:
    return str(x).strip() if x is not None else ""


def _continues_header(row: list[str]) -> bool:
    if len(row) < 2:
        return False
    label = row[0].strip()
    content = [cell.strip() for cell in row[1:] if cell.strip()]
    if not content:
        return False
    if label and not (
        _LABEL_HEADER.fullmatch(label) or _PERIOD_HEADER.fullmatch(label)
    ):
        return False
    if all(_YEAR.fullmatch(cell) for cell in content):
        return True
    return all(
        _YEAR.fullmatch(cell)
        or (parse_amount(cell)[0] is None and _PERIOD_HEADER.search(cell))
        or scale_hints(cell, declaration_only=True)
        for cell in content
    )


def parse_table_structure(table: list[list[Any]]) -> dict[str, Any]:
    if not isinstance(table, list) or not table:
        return {
            "header_rows": [],
            "header_depth": 0,
            "data_start_row": 0,
            "nested_row_labels": [],
            "has_multi_header": False,
            "has_merged_or_spanned_cells": False,
            "row_count": 0,
            "max_cols": 0,
        }

    rows = [
        (
            [str(cell) if cell is not None else "" for cell in row]
            if isinstance(row, list)
            else []
        )
        for row in table
    ]

    max_cols = max(len(r) for r in rows)

    header_depth = 1
    for row in rows[1:3]:
        if not _continues_header(row):
            break
        header_depth += 1

    header_rows = [[_clean(cell) for cell in row] for row in rows[:header_depth]]
    data_rows = rows[header_depth:]

    merged_like = False
    nested_labels: list[dict[str, Any]] = []
    last_parent = ""

    for ridx, r in enumerate(data_rows, start=header_depth):
        if not r:
            continue
        first = r[0]
        rest_has_text = any(_clean(x) for x in r[1:])
        if not first.strip() and rest_has_text:
            merged_like = True

        stripped = first.lstrip()
        indent = len(first) - len(stripped)
        if indent > 0 or stripped.startswith(("-", "*", "(", ".")):
            nested_labels.append(
                {"row_index": ridx, "label": stripped, "parent_label": last_parent}
            )
        elif stripped:
            last_parent = stripped

    return {
        "header_rows": header_rows,
        "header_depth": header_depth,
        "data_start_row": header_depth,
        "nested_row_labels": nested_labels,
        "has_multi_header": header_depth > 1,
        "has_merged_or_spanned_cells": merged_like,
        "row_count": len(rows),
        "max_cols": max_cols,
    }
