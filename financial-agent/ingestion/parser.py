"""FinQA dataset parser.

This module converts raw FinQA JSON records into a uniform parsed JSONL format
that downstream normalization, loading, and evaluation steps can consume.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

try:
	from ingestion.entity_extractor import extract_entities
	from ingestion.table_parser import parse_table_structure
	from ingestion.text_chunker import chunk_context
except ModuleNotFoundError:
	from entity_extractor import extract_entities
	from table_parser import parse_table_structure
	from text_chunker import chunk_context

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
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def token_count(text: str) -> int:
	if not isinstance(text, str) or not text.strip():
		return 0
	return len(text.split())


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
	labels: List[str] = []

	if "%" in q or "percent" in q or "percentage" in q:
		labels.append("percentage")

	diff_words = ["difference", "change", "increase", "decrease", "more", "less", "higher", "lower"]
	if any(w in q for w in diff_words) or "subtract" in program_ops:
		labels.append("difference")

	ratio_words = ["ratio", "proportion", "as a %", "fraction", "what portion"]
	if any(w in q for w in ratio_words) or "divide" in program_ops:
		labels.append("ratio")

	if len(program_ops) >= 2 or (isinstance(program, str) and "#" in program):
		labels.append("multi_hop")

	if not labels:
		labels.append("other")

	return labels


def safe_lower(x: Any) -> str:
	if x is None:
		return ""
	return str(x).strip().lower()


def detect_edge_cases(table: List[List[Any]], pre_text: List[str], post_text: List[str], program: str) -> List[str]:
	tags: List[str] = []

	if not isinstance(program, str) or not program.strip():
		tags.append("missing_program")

	if not isinstance(table, list) or not table:
		tags.append("missing_table")
		return tags

	row_lens = [len(r) for r in table if isinstance(r, list)]
	if row_lens and len(set(row_lens)) > 1:
		tags.append("uneven_row_lengths")

	merged_like = False
	missing_like = False
	years = set()
	for row in table:
		if not isinstance(row, list):
			continue
		if row and str(row[0]).strip() == "" and any(str(c).strip() for c in row[1:]):
			merged_like = True
		for c in row:
			cell = safe_lower(c)
			if cell in {"", "-", "--", "---", "na", "n/a", "none", "null", "nan", "nm"}:
				missing_like = True
			for y in YEAR_RE.findall(str(c)):
				years.add(y)

	if merged_like:
		tags.append("possible_merged_cells")
	if missing_like:
		tags.append("missing_value_cells")
	if len(years) >= 2:
		tags.append("multi_year_table")

	if (not isinstance(pre_text, list) or len(pre_text) == 0) and (not isinstance(post_text, list) or len(post_text) == 0):
		tags.append("no_context_sentences")

	return tags


@dataclass
class ParseStats:
	records: int = 0
	with_program: int = 0


def parse_record(split: str, rec: Dict[str, Any]) -> Dict[str, Any]:
	qa = rec.get("qa", {}) if isinstance(rec, dict) else {}
	question = qa.get("question", "") if isinstance(qa, dict) else ""
	program = qa.get("program", "") if isinstance(qa, dict) else ""
	program_re = qa.get("program_re", "") if isinstance(qa, dict) else ""

	table = rec.get("table", []) if isinstance(rec, dict) else []
	pre_text = rec.get("pre_text", []) if isinstance(rec, dict) else []
	post_text = rec.get("post_text", []) if isinstance(rec, dict) else []

	program_ops = extract_program_ops(program)
	question_types = classify_question(question, program_ops, program)
	edge_cases = detect_edge_cases(table, pre_text, post_text, program)
	table_structure = parse_table_structure(table)
	context_chunks = chunk_context(pre_text, post_text)
	entities = extract_entities(
		question=question,
		filename=rec.get("filename", ""),
		pre_text=pre_text,
		post_text=post_text,
		table=table,
	)

	parsed = {
		"record_id": rec.get("id"),
		"filename": rec.get("filename"),
		"split": split,
		"question": question,
		"question_token_count": token_count(question),
		"answer": qa.get("answer") if isinstance(qa, dict) else None,
		"execution_answer": qa.get("exe_ans") if isinstance(qa, dict) else None,
		"program": program,
		"program_re": program_re,
		"program_ops": program_ops,
		"question_types": question_types,
		"gold_indices": qa.get("gold_inds") if isinstance(qa, dict) else None,
		"table": table,
		"table_row_count": len(table) if isinstance(table, list) else 0,
		"table_col_count_max": max((len(r) for r in table if isinstance(r, list)), default=0),
		"pre_text": pre_text,
		"post_text": post_text,
		"pre_text_sentence_count": len(pre_text) if isinstance(pre_text, list) else 0,
		"post_text_sentence_count": len(post_text) if isinstance(post_text, list) else 0,
		"context_chunks": context_chunks,
		"table_structure": table_structure,
		"entities": entities,
		"edge_case_tags": edge_cases,
		"source": "finqa",
	}
	return parsed


def parse_split(dataset_dir: str, split: str) -> Iterable[Dict[str, Any]]:
	path = os.path.join(dataset_dir, f"{split}.json")
	with open(path, "r", encoding="utf-8") as f:
		records = json.load(f)
	for rec in records:
		if isinstance(rec, dict):
			yield parse_record(split, rec)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	count = 0
	with open(path, "w", encoding="utf-8") as f:
		for row in rows:
			f.write(json.dumps(row, ensure_ascii=True) + "\n")
			count += 1
	return count


def build_manifest(out_dir: str, split_counts: Dict[str, int], global_ops: Counter, global_qtypes: Counter) -> None:
	manifest = {
		"created_at_utc": datetime.now(timezone.utc).isoformat(),
		"format": "jsonl",
		"splits": split_counts,
		"global_program_ops": dict(global_ops),
		"global_question_types": dict(global_qtypes),
	}
	path = os.path.join(out_dir, "finqa_parsed_manifest.json")
	with open(path, "w", encoding="utf-8") as f:
		json.dump(manifest, f, indent=2)


def run(dataset_dir: str, out_dir: str, splits: List[str]) -> None:
	os.makedirs(out_dir, exist_ok=True)
	split_counts: Dict[str, int] = {}
	global_ops: Counter = Counter()
	global_qtypes: Counter = Counter()

	for split in splits:
		rows = list(parse_split(dataset_dir, split))
		split_counts[split] = len(rows)
		for r in rows:
			global_ops.update(r.get("program_ops", []))
			global_qtypes.update(r.get("question_types", []))

		out_path = os.path.join(out_dir, f"finqa_{split}_parsed.jsonl")
		write_jsonl(out_path, rows)
		print(f"wrote {len(rows)} parsed records -> {out_path}")

	build_manifest(out_dir, split_counts, global_ops, global_qtypes)
	print(f"wrote manifest -> {os.path.join(out_dir, 'finqa_parsed_manifest.json')}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Parse FinQA raw dataset into normalized JSONL-ready records")
	parser.add_argument(
		"--dataset-dir",
		default="data/raw/FinQA-main/dataset",
		help="Directory containing train/dev/test/private_test JSON",
	)
	parser.add_argument(
		"--out-dir",
		default="data/processed/parsed",
		help="Output directory for parsed JSONL files",
	)
	parser.add_argument(
		"--splits",
		nargs="+",
		default=SPLITS,
		choices=SPLITS,
		help="Dataset splits to parse",
	)
	args = parser.parse_args()

	run(args.dataset_dir, args.out_dir, args.splits)


if __name__ == "__main__":
	main()
