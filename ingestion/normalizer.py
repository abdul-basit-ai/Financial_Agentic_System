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

# Resilient imports with fallbacks
try:
    from ingestion.entity_resolution import resolve_metric_names
except (ModuleNotFoundError, ImportError):
    try:
        from entity_resolution import resolve_metric_names
    except (ModuleNotFoundError, ImportError):

        def resolve_metric_names(metrics: list[str]) -> dict[str, str]:
            return {m: m.strip().lower() for m in metrics if m}


try:
    from ingestion.unit_normalizer import normalize_with_context
except (ModuleNotFoundError, ImportError):
    try:
        from unit_normalizer import normalize_with_context
    except (ModuleNotFoundError, ImportError):

        def normalize_with_context(
            val: float | None, raw_text: str, *contexts: str
        ) -> dict[str, Any]:
            if val is None:
                return {
                    "value_base": None,
                    "unit_label": None,
                    "scale_multiplier": 1.0,
                }
            merged_context = f"{raw_text} {' '.join(contexts)}".lower()
            multiplier = 1.0
            unit_label = "unit"

            if any(w in merged_context for w in ["in billion", "billions"]):
                multiplier = 1e9
                unit_label = "billion"
            elif any(w in merged_context for w in ["in million", "millions"]):
                multiplier = 1e6
                unit_label = "million"
            elif any(w in merged_context for w in ["in thousand", "thousands"]):
                multiplier = 1e3
                unit_label = "thousand"
            elif "%" in raw_text or "percent" in merged_context:
                unit_label = "percent"

            return {
                "value_base": val * multiplier if unit_label != "percent" else val,
                "unit_label": unit_label,
                "scale_multiplier": multiplier,
            }


SPLITS = ["train", "dev", "test", "private_test"]
PER_SHARE_OR_RATIO_RE = re.compile(
    r"\b(per\s+share|per\s+common\s+share|eps|ratio|margin|percentage|rate|"
    r"shares\s+outstanding)\b",
    re.I,
)
HEADER_CONTINUATION_RE = re.compile(
    r"\b(months?\s+ended|quarter|as\s+of|year\s+ended)\b", re.I
)
UNIT_SCALE_MULTIPLIERS = {
    "thousand": 1_000.0,
    "thousands": 1_000.0,
    "million": 1_000_000.0,
    "millions": 1_000_000.0,
    "billion": 1_000_000_000.0,
    "billions": 1_000_000_000.0,
    "trillion": 1_000_000_000_000.0,
    "trillions": 1_000_000_000_000.0,
}

# Matches: $ (1,234.50)%, 123.4, (50), $45.2M, with footnotes stripped
FIN_NUMERIC_RE = re.compile(
    r"^\s*\(?\s*[\$€£¥]?\s*\(?\s*" r"(-?[0-9][0-9,]*(?:\.[0-9]+)?)\s*\)?\s*(%?)\s*$"
)
# FinQA duplicated-percent format: "11.1% ( 11.1 % )" — value restated with spaces
FIN_DUP_PCT_RE = re.compile(
    r"^\s*\(?\s*[\$€£¥]?\s*"
    r"(-?[0-9][0-9,]*(?:\.[0-9]+)?)\s*%\s*"
    r"\(\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)\s*%\s*\)\s*$"
)
FOOTNOTE_STRIP_RE = re.compile(
    r"(?<=[0-9%\)])\s*(?:\[\w+\]|\([a-zA-Z0-9*]+\)|[\*†‡#])+\s*$"
)


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def parse_numeric(value: Any) -> tuple[float | None, bool]:
    """Parses numeric accounting formats.

    Returns:
        (parsed_float, is_percentage)
    """
    if value is None:
        return None, False
    s = clean_text(value)
    if not s:
        return None, False

    # Safely strip footnotes after digits or closing accounting parentheses.
    s_cleaned = FOOTNOTE_STRIP_RE.sub("", s).strip()

    # Detect accounting negative with optional currency whitespace: $(123), € (50).
    has_opening_accounting_paren = bool(re.match(r"^[\$€£¥]?\s*\(", s_cleaned.strip()))
    is_accounting_negative = has_opening_accounting_paren and (
        s_cleaned.endswith(")") or s_cleaned.endswith(")%")
    )

    # FinQA duplicated-percent format first: "11.1% ( 11.1 % )"
    dup_match = FIN_DUP_PCT_RE.match(s_cleaned)
    if dup_match:
        num_str = dup_match.group(1).replace(",", "")
        try:
            return float(num_str), True
        except ValueError:
            return None, False

    match = FIN_NUMERIC_RE.match(s_cleaned)
    if not match:
        return None, False

    num_str, pct_token = match.group(1), match.group(2)
    num_str = num_str.replace(",", "")

    try:
        val = float(num_str)
    except ValueError:
        return None, False

    if is_accounting_negative and val > 0:
        val = -val

    is_pct = pct_token == "%" or s_cleaned.endswith("%")

    # Keep val intact (do NOT divide by 100) to preserve execution parity with FinQA exe_ans
    return val, is_pct


def normalize_cell(cell: Any, surrounding_context: str = "") -> dict[str, Any]:
    raw = clean_text(cell)
    missing_like = raw.lower() in {
        "",
        "-",
        "--",
        "---",
        "na",
        "n/a",
        "none",
        "null",
        "nan",
        "nm",
        "$ -",
        "$-",
    }
    numeric_value, is_pct = parse_numeric(raw)
    norm = normalize_with_context(numeric_value, raw, surrounding_context)
    unit_label = norm.get("unit_label")
    scale_multiplier = UNIT_SCALE_MULTIPLIERS.get(str(unit_label).lower(), 1.0)
    if is_pct:
        # Percentages are dimensionless ratios — never scale by table unit context.
        unit_label = "percent"
        scale_multiplier = 1.0
    return {
        "raw": raw,
        "is_missing": missing_like,
        "is_percentage": is_pct,
        "numeric_value": numeric_value,
        "unit_label": unit_label,
        "numeric_value_base": numeric_value if is_pct else norm["value_base"],
        "scale_multiplier": scale_multiplier,
    }


def _is_continuation_header(row: list[Any]) -> bool:
    if not isinstance(row, list):
        return False
    text_cells = [clean_text(cell) for cell in row if clean_text(cell)]
    if not text_cells:
        return False
    if HEADER_CONTINUATION_RE.search(" ".join(text_cells)):
        return True
    return all(parse_numeric(cell)[0] is None for cell in text_cells)


def normalize_table(table: list[list[Any]], text_context: str = "") -> dict[str, Any]:
    if not isinstance(table, list) or not table:
        return {
            "rows": [],
            "shape": {"rows": 0, "max_cols": 0},
            "numeric_cell_count": 0,
            "missing_cell_count": 0,
            "row_labels": [],
            "header": [],
        }

    header_rows = 1
    first_row = table[0] if isinstance(table[0], list) else []
    header_tokens = [clean_text(c) for c in first_row]
    if len(table) > 1 and _is_continuation_header(table[1]):
        header_rows = 2
        second_row = table[1]
        header_tokens = [
            " ".join(
                token
                for token in (clean_text(first_row[idx]), clean_text(second_row[idx]))
                if token
            )
            for idx in range(max(len(first_row), len(second_row)))
        ]

    # Extract row labels from column 0
    row_labels = [
        clean_text(r[0])
        for r in table[header_rows:]
        if isinstance(r, list) and len(r) > 0
    ]

    # Combine text_context (e.g. pre_text mentioning 'in millions') with table structure
    table_context = f"{text_context} {' '.join(header_tokens)} {' '.join(row_labels)}"

    rows = []
    numeric_cell_count = 0
    missing_cell_count = 0
    max_cols = 0

    for row in table[header_rows:]:
        if not isinstance(row, list):
            continue
        max_cols = max(max_cols, len(row))
        norm_row = []
        exempt_row = bool(row and PER_SHARE_OR_RATIO_RE.search(clean_text(row[0])))
        for c in row:
            nc = normalize_cell(c, table_context)
            if exempt_row and nc["numeric_value"] is not None:
                nc["scale_multiplier"] = 1.0
                nc["numeric_value_base"] = nc["numeric_value"]
                if nc["unit_label"] not in {"percent", "base"}:
                    nc["unit_label"] = "per_share"
            if nc["numeric_value"] is not None:
                numeric_cell_count += 1
            if nc["is_missing"]:
                missing_cell_count += 1
            norm_row.append(nc)
        rows.append(norm_row)

    return {
        "rows": rows,
        "shape": {"rows": len(rows), "max_cols": max_cols},
        "numeric_cell_count": numeric_cell_count,
        "missing_cell_count": missing_cell_count,
        "row_labels": row_labels,
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

    text_context = " ".join(pre_text)
    table_obj = normalize_table(rec.get("table", []), text_context=text_context)

    ans_num, ans_is_pct = parse_numeric(answer_text)
    if ans_is_pct:
        # Percentages are dimensionless — never scale by table unit context.
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
            "chunks": rec.get("context_chunks", []),
            "pre_text_sentence_count": len(pre_text),
            "post_text_sentence_count": len(post_text),
        },
        "table": table_obj,
        "table_structure": rec.get("table_structure", {}),
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
