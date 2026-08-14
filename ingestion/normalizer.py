"""Data normalizer for parsed FinQA records.

Consumes parser JSONL outputs and writes canonicalized JSONL records for
downstream graph loading and retrieval indexing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

try:
	from ingestion.entity_resolution import resolve_metric_names
	from ingestion.unit_normalizer import normalize_with_context
except ModuleNotFoundError:
	from entity_resolution import resolve_metric_names
	from unit_normalizer import normalize_with_context

SPLITS = ["train", "dev", "test", "private_test"]
NUMERIC_RE = re.compile(r"^-?[0-9][0-9,]*(?:\.[0-9]+)?%?$")


def clean_text(text: Any) -> str:
	if text is None:
		return ""
	s = str(text)
	s = s.replace("\u2013", "-")
	s = s.replace("\u2014", "-")
	s = re.sub(r"\s+", " ", s)
	return s.strip()


def parse_numeric(value: Any) -> Optional[float]:
	if value is None:
		return None
	s = clean_text(value)
	if not s:
		return None
	if not NUMERIC_RE.match(s):
		return None

	is_pct = s.endswith("%")
	if is_pct:
		s = s[:-1]

	s = s.replace(",", "")
	try:
		num = float(s)
	except ValueError:
		return None

	if is_pct:
		return num / 100.0
	return num


def normalize_cell(cell: Any) -> Dict[str, Any]:
	raw = clean_text(cell)
	missing_like = raw.lower() in {"", "-", "--", "---", "na", "n/a", "nm", "none", "null", "nan"}
	numeric_value = parse_numeric(raw)
	norm = normalize_with_context(numeric_value, raw)
	return {
		"raw": raw,
		"is_missing": missing_like,
		"numeric_value": numeric_value,
		"unit_label": norm["unit_label"],
		"numeric_value_base": norm["value_base"],
	}


def normalize_table(table: List[List[Any]]) -> Dict[str, Any]:
	if not isinstance(table, list):
		return {
			"rows": [],
			"shape": {"rows": 0, "max_cols": 0},
			"numeric_cell_count": 0,
			"missing_cell_count": 0,
			"row_labels": [],
			"header": [],
		}

	rows = []
	numeric_cell_count = 0
	missing_cell_count = 0
	max_cols = 0

	for row in table:
		if not isinstance(row, list):
			continue
		max_cols = max(max_cols, len(row))
		norm_row = []
		for c in row:
			nc = normalize_cell(c)
			if nc["numeric_value"] is not None:
				numeric_cell_count += 1
			if nc["is_missing"]:
				missing_cell_count += 1
			norm_row.append(nc)
		rows.append(norm_row)

	header = [c["raw"] for c in rows[0]] if rows else []
	row_labels = [r[0]["raw"] for r in rows[1:] if r and len(r) > 0]
	unit_context = " ".join(header + row_labels)

	for r in rows:
		for c in r:
			if c["numeric_value"] is not None:
				norm = normalize_with_context(c["numeric_value"], c["raw"], unit_context)
				c["unit_label"] = norm["unit_label"]
				c["numeric_value_base"] = norm["value_base"]

	return {
		"rows": rows,
		"shape": {"rows": len(rows), "max_cols": max_cols},
		"numeric_cell_count": numeric_cell_count,
		"missing_cell_count": missing_cell_count,
		"row_labels": row_labels,
		"header": header,
	}


def normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
	question = clean_text(rec.get("question"))
	answer_text = clean_text(rec.get("answer"))
	table_obj = normalize_table(rec.get("table", []))
	answer_norm = normalize_with_context(
		parse_numeric(answer_text),
		answer_text,
		question,
		" ".join(table_obj.get("header", [])),
	)

	pre_text = rec.get("pre_text", [])
	post_text = rec.get("post_text", [])
	pre_text = [clean_text(x) for x in pre_text] if isinstance(pre_text, list) else []
	post_text = [clean_text(x) for x in post_text] if isinstance(post_text, list) else []

	entities = rec.get("entities", {}) if isinstance(rec.get("entities"), dict) else {}
	resolved_metrics = resolve_metric_names(entities.get("metric_names", []))

	normalized = {
		"record_id": rec.get("record_id"),
		"split": rec.get("split"),
		"source": rec.get("source", "finqa"),
		"document": {
			"filename": clean_text(rec.get("filename")),
			"question": question,
			"question_token_count": rec.get("question_token_count", 0),
			"question_types": rec.get("question_types", []),
			"answer_text": answer_text,
			"answer_numeric": parse_numeric(answer_text),
			"answer_numeric_base": answer_norm["value_base"],
			"answer_unit_label": answer_norm["unit_label"],
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
	return normalized


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	count = 0
	with open(path, "w", encoding="utf-8") as f:
		for row in rows:
			f.write(json.dumps(row, ensure_ascii=True) + "\n")
			count += 1
	return count


def run(parsed_dir: str, out_dir: str, splits: List[str]) -> None:
	os.makedirs(out_dir, exist_ok=True)

	split_counts: Dict[str, int] = {}
	edge_counts: Counter = Counter()
	qtype_counts: Counter = Counter()

	for split in splits:
		in_path = os.path.join(parsed_dir, f"finqa_{split}_parsed.jsonl")
		if not os.path.exists(in_path):
			print(f"skip {split}: missing parsed input at {in_path}")
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
		print(f"wrote {count} normalized records -> {out_path}")

	summary = {
		"created_at_utc": datetime.now(timezone.utc).isoformat(),
		"splits": split_counts,
		"edge_case_tags": dict(edge_counts),
		"question_types": dict(qtype_counts),
	}
	summary_path = os.path.join(out_dir, "finqa_normalized_summary.json")
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)
	print(f"wrote summary -> {summary_path}")


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
