"""Data normalizer for parsed FinQA records.

Standardizes numeric cells, accounting negatives, currency formats, units,
and table headers without breaking DSL execution equivalence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from ingestion.entity_resolution import resolve_metric_names
from ingestion.table_parser import parse_table_structure
from ingestion.text_chunker import chunk_context
from ingestion.unit_normalizer import (
    UNIT_MULTIPLIERS,
    currency_hints,
    normalize_value,
    normalize_with_context,
    parse_amount,
    scale_hints,
)

SPLITS = ["train", "dev", "test", "private_test"]
_PER_SHARE = re.compile(r"\b(?:per\s+(?:common\s+)?share|eps)\b", re.I)
_PERCENT = re.compile(r"\bpercent(?:age)?\b|%", re.I)
_PERCENTAGE_POINTS = re.compile(r"\b(?:percent(?:age)?\s+points?|p\.p\.)\b", re.I)
_EXCEPTION = re.compile(r"\bexcept\b", re.I)
_RATIO = re.compile(r"\bratio\b", re.I)
_SHARES = re.compile(r"\bshares\b", re.I)
_MISSING = {"", "-", "--", "---", "na", "n/a", "none", "null", "nan", "nm", "$ -", "$-"}


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def parse_numeric(value: Any) -> tuple[float | None, bool]:
    number, is_percentage, _ = parse_amount(value)
    return number, is_percentage


def _first_hints(*groups: set[str]) -> set[str]:
    return next((hints for hints in groups if hints), set())


def _currency(*texts: str) -> tuple[str | None, bool]:
    tiers = [currency_hints(text) for text in texts]
    first = _first_hints(*tiers)
    if len(first) != 1:
        return None, len(first) > 1
    token = next(iter(first))
    if token not in {"$", "¥"}:
        return token, False
    family = {"USD", "CAD", "AUD", "HKD"} if token == "$" else {"JPY", "CNY"}
    for hints in tiers:
        matching = hints & family
        if matching:
            return (next(iter(matching)), False) if len(matching) == 1 else (None, True)
    return token, False


def normalize_cell(
    cell: Any,
    surrounding_context: str = "",
    *,
    row_label: str = "",
    column_header: str = "",
) -> dict[str, Any]:
    raw = str(cell) if cell is not None else ""
    cleaned = clean_text(raw)
    row_label = _EXCEPTION.split(row_label, maxsplit=1)[0]
    column_header = _EXCEPTION.split(column_header, maxsplit=1)[0]
    amount, explicit_percent, explicit_scale = parse_amount(cleaned)
    is_percentage = (
        explicit_percent
        or bool(_PERCENT.search(row_label))
        or bool(_PERCENT.search(column_header))
    )
    is_per_share = bool(
        _PER_SHARE.search(row_label) or _PER_SHARE.search(column_header)
    )
    is_ratio = bool(_RATIO.search(row_label) or _RATIO.search(column_header))
    is_shares = bool(_SHARES.search(row_label)) and not is_per_share
    scales = _first_hints(
        {explicit_scale} if explicit_scale else set(),
        scale_hints(row_label),
        scale_hints(column_header),
        scale_hints(surrounding_context, declaration_only=True),
    )
    if (is_percentage or is_per_share or is_ratio) and not explicit_scale:
        scales = set()

    currency, ambiguous_currency = _currency(
        cleaned, row_label, column_header, surrounding_context
    )
    if is_percentage or is_ratio or is_shares:
        currency, ambiguous_currency = None, False

    unit_label = next(iter(scales)) if len(scales) == 1 else "base"
    quantity_kind = "currency" if currency else "number"
    if is_shares:
        quantity_kind = "shares"
    if is_per_share:
        quantity_kind = "per_share"
        if not scales:
            unit_label = "per_share"
    if is_ratio:
        quantity_kind = "ratio"
        unit_label = "ratio"
    if is_percentage:
        quantity_kind = (
            "percentage_points"
            if _PERCENTAGE_POINTS.search(row_label)
            or _PERCENTAGE_POINTS.search(column_header)
            else "percent"
        )
        unit_label = quantity_kind

    ambiguous = (
        len(scales) > 1 or ambiguous_currency or bool(explicit_scale and is_percentage)
    )
    if ambiguous:
        unit_label = "ambiguous"

    scale_multiplier = None if ambiguous else UNIT_MULTIPLIERS.get(unit_label, 1.0)
    return {
        "raw": raw,
        "is_missing": cleaned.lower() in _MISSING,
        "is_percentage": is_percentage,
        "numeric_value": amount,
        "unit_label": unit_label,
        "quantity_kind": quantity_kind,
        "currency": currency,
        "units_ambiguous": ambiguous,
        "numeric_value_base": normalize_value(amount, unit_label),
        "scale_multiplier": scale_multiplier,
    }


def normalize_table(
    table: list[list[Any]],
    text_context: str = "",
    *,
    structure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    structure = structure if structure is not None else parse_table_structure(table)
    headers = structure["header_rows"]
    header_tokens = [
        " ".join(
            clean_text(row[column]) for row in headers if column < len(row)
        ).strip()
        for column in range(max(map(len, headers), default=0))
    ]
    header_label = header_tokens[0] if header_tokens else ""
    general_context = (
        header_label
        if scale_hints(header_label, declaration_only=True)
        else f"{header_label} {text_context}".strip()
    )
    rows = []
    row_labels = []
    row_indices = []
    numeric_count = 0
    missing_count = 0
    max_cols = 0

    data_start_row = structure["data_start_row"]
    for row_index, row in enumerate(table[data_start_row:], start=data_start_row):
        if not isinstance(row, list):
            continue
        label = clean_text(row[0]) if row else ""
        row_indices.append(row_index)
        row_labels.append(label)
        max_cols = max(max_cols, len(row))
        normalized = []
        for column, cell in enumerate(row):
            if column == 0:
                value = normalize_cell(None)
                value.update(
                    raw=str(cell) if cell is not None else "", is_missing=not label
                )
            else:
                value = normalize_cell(
                    cell,
                    general_context,
                    row_label=label,
                    column_header=(
                        header_tokens[column] if column < len(header_tokens) else ""
                    ),
                )
            numeric_count += value["numeric_value"] is not None
            missing_count += value["is_missing"]
            normalized.append(value)
        rows.append(normalized)

    return {
        "rows": rows,
        "shape": {"rows": len(rows), "max_cols": max_cols},
        "numeric_cell_count": numeric_count,
        "missing_cell_count": missing_count,
        "row_labels": row_labels,
        "source_row_indices": row_indices,
        "source_rows": [
            (
                [str(cell) if cell is not None else "" for cell in row]
                if isinstance(row, list)
                else []
            )
            for row in table
        ],
        "data_start_row": data_start_row,
        "header": header_tokens,
    }


def normalize_record(rec: dict[str, Any]) -> dict[str, Any]:
    question = clean_text(rec.get("question"))
    answer_text = clean_text(rec.get("answer"))

    pre_text = rec.get("pre_text", [])
    post_text = rec.get("post_text", [])
    pre_text = [clean_text(x) for x in pre_text] if isinstance(pre_text, list) else []
    post_text = (
        [clean_text(x) for x in post_text] if isinstance(post_text, list) else []
    )

    table = rec.get("table")
    table = table if isinstance(table, list) else []
    structure = parse_table_structure(table)
    text_context = " ".join(
        sentence
        for sentence in pre_text
        if scale_hints(sentence, declaration_only=True)
    )
    table_obj = normalize_table(table, text_context=text_context, structure=structure)

    ans_num, ans_is_pct = parse_numeric(answer_text)
    answer_norm: dict[str, str | float | None]
    if ans_is_pct:

        answer_norm = {"value_base": ans_num, "unit_label": "percent"}
    else:
        answer_norm = normalize_with_context(
            ans_num,
            answer_text,
            question,
            " ".join(table_obj.get("header", [])),
            text_context,
        )

    entities = rec.get("entities", {}) if isinstance(rec.get("entities"), dict) else {}
    resolved_metrics = resolve_metric_names(entities.get("metric_names", []))

    return {
        "record_id": rec.get("record_id"),
        "split": rec.get("split"),
        "source": rec.get("source", "finqa"),
        "document": {
            "filename": clean_text(rec.get("filename")),
            "question": question,
            "question_token_count": rec.get("question_token_count", 0),
            "question_types": rec.get("question_types", []),
            "answer_text": answer_text,
            "answer_numeric": ans_num,
            "answer_is_percentage": ans_is_pct,
            "answer_numeric_base": answer_norm["value_base"],
            "answer_unit_label": "percent" if ans_is_pct else answer_norm["unit_label"],
            "execution_answer": rec.get("execution_answer"),
        },
        "reasoning": {
            "program": clean_text(rec.get("program")),
            "program_re": clean_text(rec.get("program_re")),
            "program_ops": rec.get("program_ops", []),
            "gold_indices": rec.get("gold_indices"),
        },
        "context": {
            "pre_text": pre_text,
            "post_text": post_text,
            "chunks": chunk_context(pre_text, post_text),
            "source_indices_explicit": True,
            "pre_text_sentence_count": len(pre_text),
            "post_text_sentence_count": len(post_text),
        },
        "table": table_obj,
        "table_structure": structure,
        "entities": {
            "company_names": entities.get("company_names", []),
            "fiscal_years": entities.get("fiscal_years", []),
            "metric_names": entities.get("metric_names", []),
            "metric_names_resolved": resolved_metrics,
            "units": entities.get("units", []),
        },
        "quality": {
            "edge_case_tags": sorted(set(rec.get("edge_case_tags", []))),
            "has_program": bool(clean_text(rec.get("program"))),
        },
    }


def read_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def run(parsed_dir: str, out_dir: str, splits: list[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)

    split_counts: dict[str, int] = {}
    edge_counts: Counter = Counter()
    qtype_counts: Counter = Counter()

    for split in splits:
        in_path = os.path.join(parsed_dir, f"finqa_{split}_parsed.jsonl")
        if not os.path.exists(in_path):
            print(f"Skipping {split}: missing parsed input at {in_path}")
            continue

        normalized_rows = []
        for rec in read_jsonl(in_path):
            norm = normalize_record(rec)
            normalized_rows.append(norm)

            for tag in norm["quality"]["edge_case_tags"]:
                edge_counts[tag] += 1
            for qtype in norm["document"]["question_types"]:
                qtype_counts[qtype] += 1

        out_path = os.path.join(out_dir, f"finqa_{split}_normalized.jsonl")
        count = write_jsonl(out_path, normalized_rows)
        split_counts[split] = count
        print(f"Normalized {count} records -> {out_path}")

    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "splits": split_counts,
        "edge_case_tags": dict(edge_counts),
        "question_types": dict(qtype_counts),
    }
    summary_path = os.path.join(out_dir, "finqa_normalized_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Generated normalization summary -> {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize parsed FinQA JSONL records")
    parser.add_argument(
        "--parsed-dir",
        default="data/processed/parsed",
        help="Directory containing parser outputs",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/normalized",
        help="Directory to write normalized JSONL outputs",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=SPLITS,
        choices=SPLITS,
        help="Dataset splits to normalize",
    )
    args = parser.parse_args()
    run(args.parsed_dir, args.out_dir, args.splits)


if __name__ == "__main__":
    main()
