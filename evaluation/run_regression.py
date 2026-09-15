"""CI regression runner — evaluates the agent on a fixed FinQA sample and
fails if execution accuracy drops below the configured threshold.

Phase 11 contract (project_goals.md): callable from CI (Phase 15) that fails
below an accuracy threshold. Default mode REPLAYS recorded per-record agent
states (eval_results/states/, written by evaluation/runner.py --save-states)
through the evaluator, so CI needs no live Neo4j/pgvector. The live agent runs
ONLY when --live is passed explicitly — never as a silent fallback, because a
live run inside CI without databases would hang, and an empty fixture set must
be a loud failure, not a zero-division.

Usage:
    python evaluation/run_regression.py --min-accuracy 0.70
    python evaluation/run_regression.py --live --limit 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_live_agent():
    from agent.graph import create_financial_agent

    return create_financial_agent()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CI regression gate over the FinQA evaluation harness"
    )
    parser.add_argument("--min-accuracy", type=float, default=0.70)
    parser.add_argument(
        "--data-file",
        default="data/processed/normalized/finqa_dev_normalized.jsonl",
    )
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument(
        "--states-dir",
        default="eval_results/states",
        help="Directory of recorded per-record states written by runner.py",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run the live LangGraph agent instead of replaying recorded states "
        "(requires reachable Neo4j/pgvector).",
    )
    args = parser.parse_args()

    from evaluation.evaluator import FinQAEvaluator
    from evaluation.runner import load_finqa_records

    records = load_finqa_records(args.data_file, limit=args.limit)
    if not records:
        print(f"FAIL: no evaluation records loaded from {args.data_file}")
        return 1

    correct = 0
    evaluated = 0

    if args.live:
        evaluator = FinQAEvaluator(agent_runner=build_live_agent())
        for rec in records:
            result = evaluator.evaluate_instance(rec)
            evaluated += 1
            correct += int(result.execution_correct)
    else:
        # Replay mode: recorded baseline trajectories from the last full run.
        state_dir = Path(args.states_dir)
        if not state_dir.exists():
            print(
                f"FAIL: no recorded states directory at {state_dir}. "
                f"Run `python evaluation/runner.py --limit {args.limit}` first "
                f"to generate fixtures, or pass --live."
            )
            return 1

        evaluator = FinQAEvaluator(agent_runner=None)
        import json

        from evaluation.runner import safe_record_filename

        missing: list[str] = []
        for rec in records:
            state_file = state_dir / f"{safe_record_filename(rec.record_id)}.json"
            if not state_file.exists():
                missing.append(rec.record_id)
                continue
            with open(state_file, encoding="utf-8") as f:
                state = json.load(f)
            result = evaluator.evaluate_instance(rec, simulated_state=state)
            evaluated += 1
            correct += int(result.execution_correct)

        if evaluated == 0:
            print(
                f"FAIL: zero recorded states matched the {len(records)} selected "
                f"records in {state_dir}. Regenerate fixtures with "
                f"`python evaluation/runner.py --limit {args.limit}`, or pass --live."
            )
            return 1
        if missing:
            print(
                f"WARNING: {len(missing)}/{len(records)} records had no recorded "
                f"state and were skipped (e.g. {missing[:3]})."
            )

    if evaluated == 0:
        print("FAIL: zero records evaluated (empty fixture set?)")
        return 1

    accuracy = correct / evaluated
    print(
        f"Execution accuracy: {accuracy:.2%} over {evaluated} records "
        f"(threshold: {args.min_accuracy:.2%})"
    )

    if accuracy < args.min_accuracy:
        print(
            f"FAIL: accuracy {accuracy:.2%} below threshold {args.min_accuracy:.2%}"
        )
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
