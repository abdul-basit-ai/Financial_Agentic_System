"""Build a compact train-only FinQA program exemplar file."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def family(program: str) -> str:
    match = re.search(r"\b(table_average|add|subtract|multiply|divide)\s*\(", program)
    return match.group(1) if match else "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="data/processed/normalized/finqa_train_normalized.jsonl"
    )
    parser.add_argument("--output", default="agent/prompts/program_exemplars.json")
    parser.add_argument("--per-family", type=int, default=8)
    args = parser.parse_args()

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with open(args.input, encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            question = str(item.get("document", {}).get("question", "")).strip()
            program = str(item.get("reasoning", {}).get("program", "")).strip()
            if question and program:
                bucket = grouped[family(program)]
                if len(bucket) < args.per_family:
                    bucket.append({"question": question, "program": program})

    examples = [example for key in sorted(grouped) for example in grouped[key]]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(examples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(examples)} train-only exemplars to {output}")


if __name__ == "__main__":
    main()
