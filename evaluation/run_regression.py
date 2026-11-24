"""Run an offline candidate regression or explicitly rescore archived states."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


def _candidate_runtime() -> int:
    root = Path(__file__).resolve().parent.parent
    regression_tests = root / "tests" / "test_evaluation_regressions.py"
    if not regression_tests.is_file():
        raise ValueError(
            f"Offline candidate fixture suite is missing: {regression_tests}"
        )
    environment = {
        **os.environ,
        "FINAGENT_INTEGRATION_TESTS": "0",
        "FINAGENT_TESTS_LLM": "0",
        "OPENAI_API_KEY": "",
        "OPENROUTER_API_KEY": "",
    }
    print(
        "Executing candidate graph with deterministic providers and fixture evidence",
        flush=True,
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(regression_tests),
            "-k",
            "candidate_runtime",
        ],
        cwd=root,
        env=environment,
        check=False,
    )
    return completed.returncode


def _score_replay(
    data_file: Path, states_dir: Path, limit: int | None, min_accuracy: float
) -> int:
    from evaluation.evaluator import FinQAEvaluator
    from evaluation.runner import (
        legacy_record_filename,
        load_finqa_records,
        safe_record_filename,
    )

    if not states_dir.is_dir():
        raise ValueError(f"Saved states directory is missing: {states_dir}")
    records = load_finqa_records(str(data_file), limit=limit)
    legacy_names = Counter(
        legacy_record_filename(record.record_id) for record in records
    )
    evaluator = FinQAEvaluator()
    correct = 0
    for record in records:
        path = states_dir / f"{safe_record_filename(record.record_id)}.json"
        if path.is_file():
            wrapped = json.loads(path.read_text(encoding="utf-8"))
            if wrapped.get("record_id") != record.record_id or not isinstance(
                wrapped.get("state"), dict
            ):
                raise ValueError(
                    f"Invalid saved state envelope for record {record.record_id}"
                )
            state = wrapped["state"]
        else:
            name = legacy_record_filename(record.record_id)
            legacy = states_dir / f"{name}.json"
            if legacy_names[name] != 1 or not legacy.is_file():
                raise ValueError(
                    f"No unambiguous saved state for record {record.record_id}"
                )
            state = json.loads(legacy.read_text(encoding="utf-8"))
            if state.get("record_id") != record.record_id:
                raise ValueError(
                    f"Legacy fixture ID does not match record {record.record_id}"
                )
        if state.get("input") != record.question or "final_answer" not in state:
            raise ValueError(
                f"Saved state does not match the full case for {record.record_id}"
            )
        correct += evaluator.evaluate_instance(
            record, simulated_state=state
        ).execution_correct

    accuracy = correct / len(records)
    print(
        f"Scorer replay only: {accuracy:.2%} displayed execution accuracy over {len(records)} archived states"
    )
    return int(accuracy < min_accuracy)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed FinQA regression checks")
    parser.add_argument(
        "--mode", default="candidate", choices=("candidate", "score-replay")
    )
    parser.add_argument("--min-accuracy", type=float)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--states-dir", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    try:
        if args.mode == "candidate":
            if any(
                value is not None
                for value in (
                    args.min_accuracy,
                    args.data_file,
                    args.states_dir,
                    args.limit,
                )
            ):
                parser.error(
                    "Case selection and accuracy threshold options require --mode score-replay"
                )
            return _candidate_runtime()

        min_accuracy = args.min_accuracy if args.min_accuracy is not None else 0.70
        if not 0 <= min_accuracy <= 1:
            parser.error("min-accuracy must be between 0 and 1")
        return _score_replay(
            args.data_file
            or Path("data/processed/normalized/finqa_dev_normalized.jsonl"),
            args.states_dir or Path("eval_results/states"),
            args.limit,
            min_accuracy,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
