"""Table parsing helpers for multi-header, merged-like, and nested row labels."""

from __future__ import annotations

import re
from typing import Any

CLEAN_NUM_RE = re.compile(r"[^\d.-]")


def _clean(x: Any) -> str:
    return str(x).strip() if x is not None else ""


def _is_numeric_cell(val: str) -> bool:
    s = val.strip()
    if not s or s in {"-", "--", "---", "na", "n/a", "none", "null"}:
        return False
    # Handle accounting brackets: (123) -> -123
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    cleaned = CLEAN_NUM_RE.sub("", s)
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


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

    rows = [[_clean(c) for c in r] for r in table if isinstance(r, list)]
    if not rows:
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

    max_cols = max(len(r) for r in rows)

    header_depth = 0
    for r in rows[:3]:
        if len(r) <= 1:
            header_depth += 1
            continue
        numeric_like = sum(1 for c in r[1:] if _is_numeric_cell(c))
        # Header rows have few or no valid financial numbers
        if numeric_like <= max(1, (len(r) - 1) // 3):
            header_depth += 1
        else:
            break

    if header_depth == 0:
        header_depth = 1

    header_rows = rows[:header_depth]
    data_rows = rows[header_depth:]

    merged_like = False
    nested_labels: list[dict[str, Any]] = []
    last_parent = ""

    for ridx, r in enumerate(data_rows, start=header_depth):
        if not r:
            continue
        first = r[0]
        rest_has_text = any(_clean(x) for x in r[1:])
        if first == "" and rest_has_text:
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