"""Profile FinQA raw dataset structure, question/program types, edge cases, and metric categories.

Usage:
  /usr/bin/python3 evaluation/profile_finqa_dataset.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from statistics import mean
from typing import Dict, Iterable, List, Tuple

SPLITS = ["train", "dev", "test", "private_test"]
PROGRAM_OPS = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "exp",
    "greater",
    "table_max",
    "table_min",
    "table_sum",
    "table_average",
}
MISSING_TOKENS = {"", "-", "--", "---", "na", "n/a", "nm", "none", "null", "nan", "n.m."}
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
WORD_RE = re.compile(r"[A-Za-z0-9%\.]+")

# Keyword buckets for financial metric categories.
METRIC_CATEGORIES = {
    "revenue_sales": [
        "revenue",
        "sales",
        "net sales",
        "turnover",
        "top line",
    ],
    "profitability_income": [
        "net income",
        "operating income",
        "gross profit",
        "profit",
        "earnings",
        "ebit",
        "ebitda",
        "eps",
    ],
    "cash_flow_liquidity": [
        "cash flow",
        "cash",
        "operating activities",
        "investing activities",
        "financing activities",
        "liquidity",
    ],
    "balance_sheet": [
        "assets",
        "liabilities",
        "equity",
        "debt",
        "current assets",
        "current liabilities",
        "long-term debt",
    ],
    "cost_expense": [
        "expense",
        "cost",
        "cogs",
        "sg&a",
        "operating expenses",
        "interest expense",
        "depreciation",
        "amortization",
    ],
    "shares_capital": [
        "shares",
        "stock",
        "buyback",
        "repurchase",
        "dividend",
        "options",
        "diluted",
        "common stock",
    ],
    "tax": [
        "tax",
        "income tax",
        "effective tax rate",
    ],
    "ratio_margin": [
        "ratio",
        "margin",
        "return on",
        "roe",
        "roa",
        "percentage",
        "%",
    ],
}


def load_split(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def token_count(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(WORD_RE.findall(text))


def flatten_table(table: List[List[str]]) -> Iterable[str]:
    for row in table:
        if isinstance(row, list):
            for cell in row:
                yield str(cell)


def extract_program_ops(program: str) -> List[str]:
    if not isinstance(program, str) or not program.strip():
        return []
    ops = []
    for tok in re.split(r"[\s(),]+", program):
        t = tok.strip()
        if t in PROGRAM_OPS:
            ops.append(t)
    return ops


def classify_question(question: str, program_ops: List[str], program: str) -> List[str]:
    q = (question or "").lower()
    labels = []

    if "%" in q or "percent" in q or "percentage" in q:
        labels.append("percentage")

    diff_words = ["difference", "change", "increase", "decrease", "more", "less", "higher", "lower"]
    if any(w in q for w in diff_words) or "subtract" in program_ops:
        labels.append("difference")

    ratio_words = ["ratio", "proportion", "per ", "as a %", "fraction"]
    if any(w in q for w in ratio_words) or "divide" in program_ops:
        labels.append("ratio")

    # Multi-hop is approximated as multi-op programs and/or reference chaining.
    if len(program_ops) >= 2 or (isinstance(program, str) and "#" in program):
        labels.append("multi_hop")

    if not labels:
        labels.append("other")

    return labels


def detect_metric_categories(text: str) -> List[str]:
    t = (text or "").lower()
    matched = []
    for category, keywords in METRIC_CATEGORIES.items():
        if any(k in t for k in keywords):
            matched.append(category)
    return matched


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def profile_dataset(dataset_dir: str) -> Dict:
    top_keys_union = set()
    top_keys_intersection = None
    qa_keys_union = set()
    qa_keys_intersection = None

    split_stats = {}
    global_program_ops = Counter()
    global_qtypes = Counter()
    global_categories = Counter()

    edge_cases = Counter()

    for split in SPLITS:
        records = load_split(os.path.join(dataset_dir, f"{split}.json"))

        split_ops = Counter()
        split_qtypes = Counter()
        split_categories = Counter()

        q_lens = []
        table_rows = []
        pre_sent_counts = []
        post_sent_counts = []
        with_program = 0

        for rec in records:
            keys = set(rec.keys())
            top_keys_union.update(keys)
            top_keys_intersection = keys if top_keys_intersection is None else (top_keys_intersection & keys)

            qa = rec.get("qa", {}) if isinstance(rec, dict) else {}
            qa_keys = set(qa.keys()) if isinstance(qa, dict) else set()
            qa_keys_union.update(qa_keys)
            qa_keys_intersection = qa_keys if qa_keys_intersection is None else (qa_keys_intersection & qa_keys)

            question = qa.get("question", "") if isinstance(qa, dict) else ""
            program = qa.get("program", "") if isinstance(qa, dict) else ""
            answer = qa.get("answer", "") if isinstance(qa, dict) else ""

            q_lens.append(token_count(question))

            pre_text = rec.get("pre_text", [])
            post_text = rec.get("post_text", [])
            pre_sent_counts.append(len(pre_text) if isinstance(pre_text, list) else 0)
            post_sent_counts.append(len(post_text) if isinstance(post_text, list) else 0)

            table = rec.get("table", [])
            table_rows.append(len(table) if isinstance(table, list) else 0)

            ops = extract_program_ops(program)
            if ops:
                with_program += 1
            else:
                edge_cases[f"{split}:missing_program"] += 1

            for op in ops:
                split_ops[op] += 1
                global_program_ops[op] += 1

            qtypes = classify_question(question, ops, program)
            for qt in qtypes:
                split_qtypes[qt] += 1
                global_qtypes[qt] += 1

            # Edge cases: answer/program consistency, missing/irregular cells, year-rich headers.
            if isinstance(answer, str) and answer.strip().lower() in {"yes", "no"}:
                edge_cases[f"{split}:boolean_answer"] += 1

            if isinstance(table, list) and table:
                row_lens = [len(r) for r in table if isinstance(r, list)]
                if row_lens and len(set(row_lens)) > 1:
                    edge_cases[f"{split}:uneven_table_row_lengths"] += 1

                year_values = set()
                possible_merged_rows = 0
                missing_cells = 0

                for row in table:
                    if not isinstance(row, list):
                        continue
                    # Heuristic: blank first cell with content later often indicates merged row labels in source.
                    if row and str(row[0]).strip() == "" and any(str(c).strip() for c in row[1:]):
                        possible_merged_rows += 1

                    for c in row:
                        cell = str(c).strip().lower()
                        if cell in MISSING_TOKENS:
                            missing_cells += 1
                        for y in YEAR_RE.findall(str(c)):
                            # YEAR_RE returns prefix groups when using alternation; parse via second regex below.
                            pass
                        for full in re.findall(r"\b(?:19|20)\d{2}\b", str(c)):
                            year_values.add(full)

                if missing_cells > 0:
                    edge_cases[f"{split}:rows_with_missing_like_cells"] += 1
                if possible_merged_rows > 0:
                    edge_cases[f"{split}:possible_merged_cell_rows"] += 1
                if len(year_values) >= 2:
                    edge_cases[f"{split}:multi_year_tables"] += 1

                # Metric categories from row/header text and question.
                table_text = " ".join(flatten_table(table))
                combined_text = f"{question} {table_text}"
                cats = detect_metric_categories(combined_text)
                for c in cats:
                    split_categories[c] += 1
                    global_categories[c] += 1

            # Text retrieval edge cues.
            if isinstance(pre_text, list) and len(pre_text) == 0 and isinstance(post_text, list) and len(post_text) == 0:
                edge_cases[f"{split}:no_context_sentences"] += 1

        split_stats[split] = {
            "records": len(records),
            "records_with_program": with_program,
            "question_tokens_avg": round(mean(q_lens), 2) if q_lens else 0.0,
            "table_rows_avg": round(mean(table_rows), 2) if table_rows else 0.0,
            "pre_text_sentences_avg": round(mean(pre_sent_counts), 2) if pre_sent_counts else 0.0,
            "post_text_sentences_avg": round(mean(post_sent_counts), 2) if post_sent_counts else 0.0,
            "top_program_ops": split_ops.most_common(10),
            "question_types": dict(split_qtypes),
            "financial_metric_categories": dict(split_categories),
        }

    return {
        "dataset_dir": dataset_dir,
        "schema": {
            "top_level_keys_union": sorted(top_keys_union),
            "top_level_keys_intersection": sorted(top_keys_intersection or []),
            "qa_keys_union": sorted(qa_keys_union),
            "qa_keys_intersection": sorted(qa_keys_intersection or []),
        },
        "splits": split_stats,
        "global": {
            "program_ops": dict(global_program_ops),
            "question_types": dict(global_qtypes),
            "financial_metric_categories": dict(global_categories),
        },
        "edge_cases": dict(edge_cases),
    }


def render_markdown(summary: Dict) -> str:
    lines = []
    lines.append("# FinQA Dataset Profiling Report")
    lines.append("")
    lines.append("This report is auto-generated from raw FinQA JSON files in `data/raw/FinQA-main/dataset`.")
    lines.append("")
    lines.append("## Checklist Coverage")
    lines.append("")
    lines.append("- [x] Explore FinQA dataset structure")
    lines.append("- [x] Understand JSON schema (tables, text, questions, programs, answers)")
    lines.append("- [x] Identify question types (percentage, difference, ratio, multi-hop)")
    lines.append("- [x] Identify edge cases (missing values, merged cells, multi-year tables)")
    lines.append("- [x] Document financial metric categories found in dataset")
    lines.append("")

    schema = summary["schema"]
    lines.append("## JSON Schema Summary")
    lines.append("")
    lines.append("### Top-level keys (intersection across splits)")
    lines.append("")
    for k in schema["top_level_keys_intersection"]:
        lines.append(f"- {k}")
    lines.append("")
    lines.append("### QA keys (intersection across splits)")
    lines.append("")
    for k in schema["qa_keys_intersection"]:
        lines.append(f"- {k}")
    lines.append("")

    lines.append("## Split Statistics")
    lines.append("")
    for split, s in summary["splits"].items():
        lines.append(f"### {split}")
        lines.append("")
        lines.append(f"- Records: {s['records']}")
        lines.append(f"- Records with program: {s['records_with_program']}")
        lines.append(f"- Avg question token count: {s['question_tokens_avg']}")
        lines.append(f"- Avg table rows: {s['table_rows_avg']}")
        lines.append(f"- Avg pre-text sentences: {s['pre_text_sentences_avg']}")
        lines.append(f"- Avg post-text sentences: {s['post_text_sentences_avg']}")
        lines.append("- Top program ops:")
        for op, c in s["top_program_ops"][:8]:
            lines.append(f"  - {op}: {c}")
        lines.append("- Question types:")
        for qt, c in sorted(s["question_types"].items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"  - {qt}: {c}")
        lines.append("- Financial metric categories:")
        for cat, c in sorted(s["financial_metric_categories"].items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"  - {cat}: {c}")
        lines.append("")

    lines.append("## Edge Cases")
    lines.append("")
    for name, count in sorted(summary["edge_cases"].items(), key=lambda kv: kv[0]):
        lines.append(f"- {name}: {count}")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- `private_test` has no gold programs by design (blind evaluation split).")
    lines.append("- `possible_merged_cell_rows` is a heuristic: rows with blank first cell and populated trailing cells.")
    lines.append("- Question type detection is heuristic and should be refined in the parser for training/eval tasks.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile FinQA raw dataset")
    parser.add_argument(
        "--dataset-dir",
        default="data/raw/FinQA-main/dataset",
        help="Directory containing train/dev/test/private_test JSON files",
    )
    parser.add_argument(
        "--output-json",
        default="data/processed/finqa_dataset_profile.json",
        help="Path to write machine-readable summary JSON",
    )
    parser.add_argument(
        "--output-md",
        default="evaluation/FINQA_DATASET_PROFILE.md",
        help="Path to write markdown report",
    )
    args = parser.parse_args()

    summary = profile_dataset(args.dataset_dir)

    ensure_parent(args.output_json)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md = render_markdown(summary)
    ensure_parent(args.output_md)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Wrote JSON summary: {args.output_json}")
    print(f"Wrote Markdown report: {args.output_md}")


if __name__ == "__main__":
    main()
