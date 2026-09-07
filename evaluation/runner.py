"""CLI benchmark runner executing evaluation batches over FinQA splits."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from agent.graph import create_financial_agent
from evaluation.evaluator import EvaluationRecord, EvaluationResult, FinQAEvaluator


def load_finqa_records(filepath: str, limit: int | None = None) -> list[EvaluationRecord]:
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
            # Gold evidence IDs are the KEYS of gold_indices (table_row_N / pre_text etc.)
            gold_inds = [str(k) for k in gold_indices.keys()]

            company = None
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
    parser = argparse.ArgumentParser(description="Run Phase 11 FinQA Evaluation Benchmark")
    parser.add_argument("--data-file", default="data/processed/normalized/finqa_dev_normalized.jsonl")
    parser.add_argument("--output-dir", default="eval_results")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    records = load_finqa_records(args.data_file, limit=args.limit)

    agent_runner = create_financial_agent()
    evaluator = FinQAEvaluator(agent_runner=agent_runner)

    results: list[EvaluationResult] = []
    print(f"Executing benchmark on {len(records)} records...")

    for i, rec in enumerate(records, start=1):
        res = evaluator.evaluate_instance(rec)
        results.append(res)
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

    # Save Artifacts
    json_path = os.path.join(args.output_dir, "benchmark_summary.json")
    md_path = os.path.join(args.output_dir, "benchmark_report.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(generate_markdown_report(summary))

    print(f"\nBenchmark Complete! Reports written to:\n  - {json_path}\n  - {md_path}")
    print(f"Final Acc_exe: {exe_acc:.2%} | Program Acc: {prog_acc:.2%}")


if __name__ == "__main__":
    main()