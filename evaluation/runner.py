"""CLI benchmark runner executing evaluation batches over FinQA splits."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from agent.graph import create_financial_agent
from evaluation.evaluator import EvaluationRecord, EvaluationResult, FinQAEvaluator
from ingestion.entity_extractor import extract_company_identifier


def safe_record_filename(record_id: str) -> str:
    """Filesystem-safe fixture filename stem for a FinQA record id.

    FinQA record ids are path-like ('V/2008/page_17.pdf-1'), so they must be
    flattened before use as a filename. run_regression.py must use this SAME
    function when reading fixtures back.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(record_id))


def load_finqa_records(
    filepath: str, limit: int | None = None
) -> list[EvaluationRecord]:
    """Loads normalized FinQA records. Field paths follow the Phase 2 schema:
    document.execution_answer, reasoning.program, reasoning.gold_indices."""
    records: list[EvaluationRecord] = []
    with open(filepath, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            doc = item.get("document", {})
            reasoning = item.get("reasoning", {})
            gold_indices = reasoning.get("gold_indices") or {}
            # Gold evidence IDs are the KEYS of gold_indices (table_row_N /
            # pre_text etc.); the VALUES are the human-readable evidence
            # strings retrieval is actually graded against (content overlap).
            gold_inds = [str(k) for k in gold_indices.keys()]
            gold_evidence = [str(v) for v in gold_indices.values() if str(v).strip()]

            # The filename ticker IS the filer (FinQA ground truth).
            # company_names mixes in-text mentions with the ticker and is
            # alphabetically sorted — company_names[0] can be a company the
            # narrative merely compares against, anchoring retrieval on the
            # wrong firm.
            company = extract_company_identifier(str(doc.get("filename", "")))
            if not company:
                entities = item.get("entities", {})
                company_list = entities.get("company_names") or []
                if company_list:
                    company = company_list[0]

            gold_answer = doc.get("execution_answer")
            if gold_answer is None:
                gold_answer = doc.get("answer_text", 0.0)

            records.append(
                EvaluationRecord(
                    record_id=item.get("record_id", f"rec_{idx}"),
                    question=doc.get("question", ""),
                    company_identifier=company,
                    gold_answer=gold_answer,
                    gold_program=reasoning.get("program"),
                    gold_inds=gold_inds,
                    gold_evidence=gold_evidence,
                    split=item.get("split", "dev"),
                )
            )
    return records


def generate_markdown_report(summary: dict[str, Any]) -> str:
    taxonomy_table = "\n".join(
        f"| {cat} | {count} | {count / summary['total_samples']:.1%} |"
        for cat, count in summary["taxonomy_breakdown"].items()
    )

    return f"""# Autonomous Financial Reasoning Agent - Benchmark Report

**Dataset Split:** {summary['split']}

**Total Samples:** {summary['total_samples']}

**Execution Accuracy (Acc_exe):** {summary['execution_accuracy']:.2%}

**Program Accuracy (Acc_prog):** {summary['program_accuracy']:.2%}

## Information Retrieval (IR) Multi-Hop Metrics
* **Recall@5:** {summary['mean_recall_at_5']:.2%}
* **Precision@5:** {summary['mean_precision_at_5']:.2%}
* **Mean Reciprocal Rank (MRR):** {summary['mean_mrr']:.4f}
* **NDCG@5:** {summary['mean_ndcg_at_5']:.4f}

## Latency & Performance
* **Mean Wall Latency:** {summary['mean_latency_ms']:.2f} ms
* **P95 Latency:** {summary['p95_latency_ms']:.2f} ms

## Causal Failure Taxonomy
| Category | Count | Proportion |
| :--- | :---: | :---: |
{taxonomy_table}
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 11 FinQA Evaluation Benchmark"
    )
    parser.add_argument(
        "--data-file", default="data/processed/normalized/finqa_dev_normalized.jsonl"
    )
    parser.add_argument("--output-dir", default="eval_results")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--save-states",
        dest="save_states",
        action="store_true",
        default=True,
        help="Write each record's final agent state to <output-dir>/states/ "
        "so run_regression.py can replay it without live databases (default on).",
    )
    parser.add_argument(
        "--no-save-states",
        dest="save_states",
        action="store_false",
        help="Skip writing per-record state files.",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    states_dir = Path(args.output_dir) / "states"
    if args.save_states:
        states_dir.mkdir(parents=True, exist_ok=True)

    records = load_finqa_records(args.data_file, limit=args.limit)

    agent_runner = create_financial_agent()
    evaluator = FinQAEvaluator(agent_runner=agent_runner)

    results: list[EvaluationResult] = []
    print(f"Executing benchmark on {len(records)} records...")

    for i, rec in enumerate(records, start=1):
        res = evaluator.evaluate_instance(rec)
        results.append(res)

        if args.save_states:
            # Persist the trajectory the agent actually took for this record
            # so the CI regression gate can replay it deterministically.
            # Must use the EVALUATOR's run-scoped thread id — the checkpointer
            # keeps every historical thread, so a bare eval_{record_id} lookup
            # would return a stale trajectory from an earlier benchmark run.
            config = {"configurable": {"thread_id": evaluator.thread_id_for(rec)}}
            try:
                final_state = agent_runner.get_state(config)
                state_values = final_state.values if final_state else {}
            except Exception:
                state_values = {}
            if state_values:
                state_path = states_dir / f"{safe_record_filename(rec.record_id)}.json"
                with open(state_path, "w", encoding="utf-8") as f:
                    json.dump(state_values, f, ensure_ascii=False, default=str)

        if i % 10 == 0 or i == len(records):
            print(f"Processed {i}/{len(records)} instances...")

    # Aggregate Metrics
    n = len(results)
    exe_acc = sum(1 for r in results if r.execution_correct) / n if n else 0.0
    prog_acc = sum(1 for r in results if r.program_correct) / n if n else 0.0
    mean_recall = sum(r.ir_metrics["recall_at_k"] for r in results) / n if n else 0.0
    mean_prec = sum(r.ir_metrics["precision_at_k"] for r in results) / n if n else 0.0
    mean_mrr = sum(r.ir_metrics["mrr"] for r in results) / n if n else 0.0
    mean_ndcg = sum(r.ir_metrics["ndcg_at_k"] for r in results) / n if n else 0.0

    latencies = sorted(r.latency_ms for r in results)
    mean_lat = sum(latencies) / n if n else 0.0
    p95_lat = latencies[int(n * 0.95)] if n else 0.0

    taxonomy_counts = dict(Counter(r.taxonomy.value for r in results))

    summary = {
        "split": records[0].split if records else "unknown",
        "total_samples": n,
        "execution_accuracy": round(exe_acc, 4),
        "program_accuracy": round(prog_acc, 4),
        "mean_recall_at_5": round(mean_recall, 4),
        "mean_precision_at_5": round(mean_prec, 4),
        "mean_mrr": round(mean_mrr, 4),
        "mean_ndcg_at_5": round(mean_ndcg, 4),
        "mean_latency_ms": round(mean_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "taxonomy_breakdown": taxonomy_counts,
    }

    # Save Artifacts — summary for humans, full per-record results for the
    # statistical regression gate (evaluation/gate.py expects the "results"
    # list with record_id + execution_correct fields), and per-record states
    # for the CI regression replay (run_regression.py).
    json_path = os.path.join(args.output_dir, "benchmark_summary.json")
    results_path = os.path.join(args.output_dir, "benchmark_results.json")
    md_path = os.path.join(args.output_dir, "benchmark_report.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": summary["split"],
                "total_samples": summary["total_samples"],
                "execution_accuracy": summary["execution_accuracy"],
                "results": [r.model_dump() for r in results],
            },
            f,
            indent=2,
            default=str,
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(generate_markdown_report(summary))

    print(
        f"\nBenchmark Complete! Reports written to:\n  - {json_path}\n  - {results_path}\n  - {md_path}"
    )
    print(f"Final Acc_exe: {exe_acc:.2%} | Program Acc: {prog_acc:.2%}")


if __name__ == "__main__":
    main()
