"""Validate parsed FinQA records against raw JSON and produce quality reports."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple

SPLITS = ["train", "dev", "test", "private_test"]


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_raw_split(dataset_dir: str, split: str) -> List[Dict[str, Any]]:
    path = os.path.join(dataset_dir, f"{split}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_raw_records(raw_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for rec in raw_records:
        rid = str(rec.get("id", ""))
        if rid:
            out[rid] = rec
    return out


def missing_like(x: Any) -> bool:
    v = str(x).strip().lower() if x is not None else ""
    return v in {"", "-", "--", "---", "na", "n/a", "nm", "none", "null", "nan"}


def compare_record(parsed: Dict[str, Any], raw: Dict[str, Any], split: str) -> Tuple[List[str], Counter, Dict[str, float]]:
    anomalies: List[str] = []
    missing = Counter()

    qa = raw.get("qa", {}) if isinstance(raw.get("qa"), dict) else {}

    # Core field equality checks.
    if str(parsed.get("filename", "")) != str(raw.get("filename", "")):
        anomalies.append("filename_mismatch")
    if str(parsed.get("question", "")) != str(qa.get("question", "")):
        anomalies.append("question_mismatch")
    raw_answer = str(qa.get("answer", ""))
    parsed_answer = str(parsed.get("answer", ""))
    # private_test is unlabeled in FinQA by design; avoid treating missing answers as anomalies.
    if split != "private_test" and parsed_answer != raw_answer:
        anomalies.append("answer_mismatch")
    raw_program = str(qa.get("program", ""))
    parsed_program = str(parsed.get("program", ""))
    # private_test is unlabeled in FinQA by design; avoid treating missing programs as anomalies.
    if split != "private_test" and parsed_program != raw_program:
        anomalies.append("program_mismatch")

    # Shape and count checks.
    raw_table = raw.get("table", []) if isinstance(raw.get("table"), list) else []
    parsed_table = parsed.get("table", []) if isinstance(parsed.get("table"), list) else []
    if len(parsed_table) != len(raw_table):
        anomalies.append("table_row_count_mismatch")

    if int(parsed.get("pre_text_sentence_count", 0)) != len(raw.get("pre_text", []) if isinstance(raw.get("pre_text"), list) else []):
        anomalies.append("pre_text_count_mismatch")
    if int(parsed.get("post_text_sentence_count", 0)) != len(raw.get("post_text", []) if isinstance(raw.get("post_text"), list) else []):
        anomalies.append("post_text_count_mismatch")

    # Missing/null analysis.
    if not str(qa.get("program", "")).strip():
        missing["missing_program"] += 1
    if not str(qa.get("answer", "")).strip():
        missing["missing_answer"] += 1
    if not str(qa.get("question", "")).strip():
        missing["missing_question"] += 1
    if not raw_table:
        missing["missing_table"] += 1

    missing_cells = 0
    total_cells = 0
    for row in raw_table:
        if not isinstance(row, list):
            continue
        for c in row:
            total_cells += 1
            if missing_like(c):
                missing_cells += 1

    pct = {
        "table_missing_cell_pct": (missing_cells / total_cells * 100.0) if total_cells else 0.0,
        "has_multi_year_table": 1.0 if any(
            any(str(c).strip().isdigit() and len(str(c).strip()) == 4 for c in row if isinstance(row, list))
            for row in raw_table
        ) else 0.0,
    }

    # Edge tags consistency checks.
    edge_tags = set(parsed.get("edge_case_tags", []))
    if missing["missing_program"] > 0 and "missing_program" not in edge_tags:
        anomalies.append("edge_tag_missing_program_not_set")

    return anomalies, missing, pct


def validate_split(dataset_dir: str, parsed_dir: str, split: str) -> Dict[str, Any]:
    raw_records = load_raw_split(dataset_dir, split)
    raw_by_id = index_raw_records(raw_records)

    parsed_path = os.path.join(parsed_dir, f"finqa_{split}_parsed.jsonl")
    parsed_records = list(read_jsonl(parsed_path)) if os.path.exists(parsed_path) else []

    coverage = {
        "raw_records": len(raw_records),
        "parsed_records": len(parsed_records),
        "parsed_coverage_pct": (len(parsed_records) / len(raw_records) * 100.0) if raw_records else 0.0,
    }

    anomalies = Counter()
    missing = Counter()
    missing_pcts = defaultdict(list)

    parsed_ids = set()
    for rec in parsed_records:
        rid = str(rec.get("record_id", ""))
        if not rid:
            anomalies["parsed_missing_record_id"] += 1
            continue
        parsed_ids.add(rid)

        raw = raw_by_id.get(rid)
        if raw is None:
            anomalies["parsed_record_not_in_raw"] += 1
            continue

        local_anoms, local_missing, local_pct = compare_record(rec, raw, split)
        for a in local_anoms:
            anomalies[a] += 1
        missing.update(local_missing)
        for k, v in local_pct.items():
            missing_pcts[k].append(v)

    for rid in raw_by_id:
        if rid not in parsed_ids:
            anomalies["raw_record_missing_in_parsed"] += 1

    summary_pct = {k: round(mean(vals), 4) if vals else 0.0 for k, vals in missing_pcts.items()}

    # Global missing percentages over records.
    record_count = max(1, len(raw_records))
    missing_pct = {f"{k}_pct": round(v / record_count * 100.0, 4) for k, v in missing.items()}

    return {
        "split": split,
        "coverage": coverage,
        "missing_counts": dict(missing),
        "missing_percentages": missing_pct,
        "quality_percentages": summary_pct,
        "anomalies": dict(anomalies),
    }


def render_markdown(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("# FinQA Data Quality Report")
    lines.append("")
    lines.append(f"Generated at: {report['created_at_utc']}")
    lines.append("")
    lines.append("## Validation Scope")
    lines.append("")
    lines.append("- Parsed records validated against original raw JSON by record id")
    lines.append("- Field parity checks for filename/question/answer/program")
    lines.append("- Missing/null analysis and anomaly counts")
    lines.append("")

    lines.append("## Split Results")
    lines.append("")
    for split_data in report["splits"]:
        split = split_data["split"]
        cov = split_data["coverage"]
        lines.append(f"### {split}")
        lines.append("")
        lines.append(f"- Raw records: {cov['raw_records']}")
        lines.append(f"- Parsed records: {cov['parsed_records']}")
        lines.append(f"- Coverage %: {cov['parsed_coverage_pct']:.4f}")

        lines.append("- Missing counts:")
        for k, v in sorted(split_data["missing_counts"].items()):
            lines.append(f"  - {k}: {v}")

        lines.append("- Missing percentages:")
        for k, v in sorted(split_data["missing_percentages"].items()):
            lines.append(f"  - {k}: {v:.4f}")

        lines.append("- Quality percentages:")
        for k, v in sorted(split_data["quality_percentages"].items()):
            lines.append(f"  - {k}: {v:.4f}")

        lines.append("- Anomalies:")
        if split_data["anomalies"]:
            for k, v in sorted(split_data["anomalies"].items()):
                lines.append(f"  - {k}: {v}")
        else:
            lines.append("  - none")
        lines.append("")

    return "\n".join(lines)


def run(dataset_dir: str, parsed_dir: str, out_json: str, out_md: str, splits: List[str]) -> Dict[str, Any]:
    split_reports = [validate_split(dataset_dir, parsed_dir, split) for split in splits]

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": dataset_dir,
        "parsed_dir": parsed_dir,
        "splits": split_reports,
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(render_markdown(report))

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate parsed FinQA records and generate quality reports")
    parser.add_argument("--dataset-dir", default="data/raw/FinQA-main/dataset")
    parser.add_argument("--parsed-dir", default="data/processed/parsed")
    parser.add_argument("--out-json", default="data/processed/finqa_data_quality_report.json")
    parser.add_argument("--out-md", default="evaluation/FINQA_DATA_QUALITY_REPORT.md")
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    args = parser.parse_args()

    report = run(
        dataset_dir=args.dataset_dir,
        parsed_dir=args.parsed_dir,
        out_json=args.out_json,
        out_md=args.out_md,
        splits=args.splits,
    )

    for s in report["splits"]:
        cov = s["coverage"]
        print(
            f"{s['split']}: coverage={cov['parsed_coverage_pct']:.2f}% "
            f"raw={cov['raw_records']} parsed={cov['parsed_records']} anomalies={sum(s['anomalies'].values())}"
        )


if __name__ == "__main__":
    main()
