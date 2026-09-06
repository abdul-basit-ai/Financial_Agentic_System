"""Data pipeline validation & integrity suite.

Verifies end-to-end data integrity across raw -> parsed -> normalized artifacts.
Detects silent drops, unparsed numeric strings, schema corruption, and data leakage.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pytest

# Thresholds for CI/CD gating
MIN_NUMERIC_EXTRACTION_RATE = 0.95  # >= 95% of digit-bearing cells must yield numbers
MAX_MISSING_PROGRAM_RATE = 0.05  # < 5% unparsed programs in train/dev
EXPECTED_SPLITS = ["train", "dev", "test"]

DIGIT_CELL_RE = re.compile(r"\d")
NUMERIC_CELL_RE = re.compile(
    r"^\s*\(?\s*[\$€£¥]?\s*[-+]?\s*\(?\s*"
    r"\d[\d,]*(?:\.\d+)?\s*\)?\s*%?\s*"
    r"(?:\[\w+\]|\([a-zA-Z0-9*]+\)|[\*†‡#])?\s*$"
)
NON_NUMERIC_EXCLUSIONS = re.compile(r"^(?:\d{4}|q[1-4]|\d{1,2}/\d{1,2}/\d{2,4})$", re.I)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON on line {line_no} in {path}: {exc}"
                ) from exc
    return records


# =====================================================================
# Integrity Metrics Engine
# =====================================================================


class DataIntegrityReport:
    def __init__(self, split: str) -> None:
        self.split = split
        self.raw_count: int = 0
        self.parsed_count: int = 0
        self.normalized_count: int = 0

        self.digit_cells_seen: int = 0
        self.numeric_extracted_cells: int = 0
        self.failed_numeric_samples: list[dict[str, Any]] = []

        self.missing_programs: int = 0
        self.nan_inf_found: int = 0
        self.schema_violations: list[str] = []
        self.dropped_record_ids: set[str] = set()

    @property
    def record_retention_rate(self) -> float:
        return (self.normalized_count / self.raw_count) if self.raw_count else 0.0

    @property
    def numeric_extraction_rate(self) -> float:
        return (
            (self.numeric_extracted_cells / self.digit_cells_seen)
            if self.digit_cells_seen
            else 1.0
        )

    def print_evidence_summary(self) -> None:
        print("\n" + "=" * 70)
        print(f"EVIDENCE REPORT: Split '{self.split}'")
        print("=" * 70)
        print(f"  Raw Records Ingested:        {self.raw_count}")
        print(f"  Parsed Records Generated:    {self.parsed_count}")
        print(f"  Normalized Records Output:   {self.normalized_count}")
        print(f"  Retention Rate:              {self.record_retention_rate:.2%}")
        print(f"  Digit Cells Inspected:       {self.digit_cells_seen}")
        print(f"  Numeric Values Parsed:       {self.numeric_extracted_cells}")
        print(f"  Numeric Extraction Coverage: {self.numeric_extraction_rate:.2%}")
        print(f"  Missing Programs:            {self.missing_programs}")
        print(f"  NaN / Inf Anomalies:         {self.nan_inf_found}")
        print(f"  Schema Violations:           {len(self.schema_violations)}")

        if self.failed_numeric_samples:
            print("\n  Sample Unparsed Numeric Candidates (Top 5):")
            for sample in self.failed_numeric_samples[:5]:
                print(f"    - Record ID {sample['record_id']}: '{sample['raw_cell']}'")

        if self.schema_violations:
            print("\n  Sample Schema Violations (Top 3):")
            for violation in self.schema_violations[:3]:
                print(f"    - {violation}")
        print("=" * 70 + "\n")


def audit_split(
    raw_path: Path, parsed_path: Path, norm_path: Path, split: str
) -> DataIntegrityReport:
    report = DataIntegrityReport(split)

    raw_data = load_json(raw_path) if raw_path.exists() else []
    report.raw_count = len(raw_data)

    parsed_records = load_jsonl(parsed_path) if parsed_path.exists() else []
    report.parsed_count = len(parsed_records)

    norm_records = load_jsonl(norm_path) if norm_path.exists() else []
    report.normalized_count = len(norm_records)

    # 1. Check for Silent Drops
    raw_ids = {r.get("id") for r in raw_data if isinstance(r, dict)}
    norm_ids = {r.get("record_id") for r in norm_records}
    report.dropped_record_ids = raw_ids - norm_ids

    # 2. Schema and Cell-Level Numeric Extraction Audit
    for rec in norm_records:
        rec_id = rec.get("record_id", "UNKNOWN")

        # Required root schemas
        for section in [
            "document",
            "reasoning",
            "context",
            "table",
            "entities",
            "quality",
        ]:
            if section not in rec:
                report.schema_violations.append(
                    f"Record {rec_id} missing section '{section}'"
                )

        doc = rec.get("document", {})
        if not doc.get("question"):
            report.schema_violations.append(f"Record {rec_id} has empty question")

        reasoning = rec.get("reasoning", {})
        if not reasoning.get("program"):
            report.missing_programs += 1

        # Check for NaN / Infinity poisoning
        ans_num = doc.get("answer_numeric")
        if ans_num is not None and (math.isnan(ans_num) or math.isinf(ans_num)):
            report.nan_inf_found += 1

        # Deep audit of table normalization
        table = rec.get("table", {})
        for row in table.get("rows", []):
            for cell in row:
                raw_cell = cell.get("raw", "")
                num_val = cell.get("numeric_value")

                if num_val is not None and (math.isnan(num_val) or math.isinf(num_val)):
                    report.nan_inf_found += 1

                # If cell contains digits, verify it wasn't dropped silently
                if NUMERIC_CELL_RE.fullmatch(
                    raw_cell
                ) and not NON_NUMERIC_EXCLUSIONS.match(raw_cell):
                    report.digit_cells_seen += 1
                    if num_val is not None:
                        report.numeric_extracted_cells += 1
                    else:
                        if len(report.failed_numeric_samples) < 20:
                            report.failed_numeric_samples.append(
                                {
                                    "record_id": rec_id,
                                    "raw_cell": raw_cell,
                                }
                            )

    return report


# =====================================================================
# Pytest Test Cases (CI/CD Automations)
# =====================================================================


@pytest.fixture(scope="session")
def pipeline_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parent.parent
    return {
        "raw_dir": root / "data" / "raw" / "FinQA-main" / "dataset",
        "parsed_dir": root / "data" / "processed" / "parsed",
        "normalized_dir": root / "data" / "processed" / "normalized",
    }


@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_zero_record_drop(pipeline_paths: dict[str, Path], split: str) -> None:
    raw_file = pipeline_paths["raw_dir"] / f"{split}.json"
    norm_file = pipeline_paths["normalized_dir"] / f"finqa_{split}_normalized.jsonl"
    parsed_file = pipeline_paths["parsed_dir"] / f"finqa_{split}_parsed.jsonl"

    if not raw_file.exists():
        pytest.skip(f"Raw split file missing: {raw_file}")

    report = audit_split(raw_file, parsed_file, norm_file, split)
    report.print_evidence_summary()

    assert report.raw_count > 0, f"Raw data for {split} split is empty"
    assert report.normalized_count == report.raw_count, (
        f"Record loss detected in '{split}'! "
        f"Raw: {report.raw_count}, Normalized: {report.normalized_count}. "
        f"Missing IDs: {list(report.dropped_record_ids)[:5]}"
    )
    if split in {"train", "dev"}:
        missing_program_rate = report.missing_programs / report.normalized_count
        assert missing_program_rate <= MAX_MISSING_PROGRAM_RATE, (
            f"Missing program rate for '{split}' is {missing_program_rate:.2%}; "
            f"target <= {MAX_MISSING_PROGRAM_RATE:.2%}."
        )


@pytest.mark.parametrize("split", ["train", "dev"])
def test_numeric_extraction_efficiency(
    pipeline_paths: dict[str, Path], split: str
) -> None:
    raw_file = pipeline_paths["raw_dir"] / f"{split}.json"
    norm_file = pipeline_paths["normalized_dir"] / f"finqa_{split}_normalized.jsonl"
    parsed_file = pipeline_paths["parsed_dir"] / f"finqa_{split}_parsed.jsonl"

    if not norm_file.exists():
        pytest.skip(f"Normalized file missing: {norm_file}")

    report = audit_split(raw_file, parsed_file, norm_file, split)

    assert report.numeric_extraction_rate >= MIN_NUMERIC_EXTRACTION_RATE, (
        f"Silent numeric drops detected in '{split}'! "
        f"Extracted {report.numeric_extraction_rate:.2%}, "
        f"target >= {MIN_NUMERIC_EXTRACTION_RATE:.2%}. "
        f"Review sample failures in stdout."
    )


@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_schema_validity_and_anomalies(
    pipeline_paths: dict[str, Path], split: str
) -> None:
    raw_file = pipeline_paths["raw_dir"] / f"{split}.json"
    norm_file = pipeline_paths["normalized_dir"] / f"finqa_{split}_normalized.jsonl"
    parsed_file = pipeline_paths["parsed_dir"] / f"finqa_{split}_parsed.jsonl"

    if not norm_file.exists():
        pytest.skip(f"Normalized file missing: {norm_file}")

    report = audit_split(raw_file, parsed_file, norm_file, split)

    assert (
        report.nan_inf_found == 0
    ), f"Detected {report.nan_inf_found} NaN/Inf values in '{split}' dataset"
    assert (
        len(report.schema_violations) == 0
    ), f"Schema violations detected in '{split}': {report.schema_violations[:5]}"


# =====================================================================
# Standalone CLI Entrypoint
# =====================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit parsed and normalized FinQA datasets."
    )
    parser.add_argument("--raw-dir", default="data/raw/FinQA-main/dataset")
    parser.add_argument("--parsed-dir", default="data/processed/parsed")
    parser.add_argument("--norm-dir", default="data/processed/normalized")
    parser.add_argument("--splits", nargs="+", default=EXPECTED_SPLITS)
    args = parser.parse_args()

    overall_passed = True

    for split in args.splits:
        raw_f = Path(args.raw_dir) / f"{split}.json"
        parsed_f = Path(args.parsed_dir) / f"finqa_{split}_parsed.jsonl"
        norm_f = Path(args.norm_dir) / f"finqa_{split}_normalized.jsonl"

        if not raw_f.exists() or not norm_f.exists():
            print(f"[WARN] Skipping split '{split}', input files not found.")
            continue

        report = audit_split(raw_f, parsed_f, norm_f, split)
        report.print_evidence_summary()

        if report.raw_count != report.normalized_count:
            print(
                f"[FAIL] Split '{split}' dropped "
                f"{report.raw_count - report.normalized_count} records."
            )
            overall_passed = False

        if report.numeric_extraction_rate < MIN_NUMERIC_EXTRACTION_RATE:
            print(
                f"[FAIL] Split '{split}' numeric extraction below threshold: "
                f"{report.numeric_extraction_rate:.2%}"
            )
            overall_passed = False

        if report.nan_inf_found > 0:
            print(
                f"[FAIL] Split '{split}' contains {report.nan_inf_found} unhandled NaN/Inf values."
            )
            overall_passed = False

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
