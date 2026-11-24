"""CLI benchmark runner executing evaluation batches over FinQA splits."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

from evaluation.evaluator import EvaluationRecord, EvaluationResult, FinQAEvaluator
from evaluation.manifest import (
    RunConfig,
    RunManifest,
    code_digest,
    hash_file,
    hash_json,
    id_set_digest,
)
from evaluation.metrics import execution_value
from ingestion.entity_extractor import extract_company_identifier


def legacy_record_filename(record_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", record_id)


def safe_record_filename(record_id: str) -> str:
    suffix = sha256(record_id.encode()).hexdigest()
    return f"{legacy_record_filename(record_id)[:80]}-{suffix}"


def load_finqa_records(
    filepath: str, limit: int | None = None, *, require_current_provenance: bool = False
) -> list[EvaluationRecord]:
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive")

    records: list[EvaluationRecord] = []
    ids: list[str] = []
    with open(filepath, encoding="utf-8") as source:
        for number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            record_id = item.get("record_id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"Missing record ID on line {number}")
            ids.append(record_id)
            if limit is not None and len(records) >= limit:
                continue

            doc = item.get("document", {})
            reasoning = item.get("reasoning", {})
            gold_indices = reasoning.get("gold_indices") or {}
            company: str | None = extract_company_identifier(
                str(doc.get("filename", ""))
            )
            if not company:
                names = item.get("entities", {}).get("company_names") or []
                company = names[0] if names else None

            gold_answer = doc.get("execution_answer")
            if execution_value(gold_answer) is None:
                raise ValueError(
                    f"Record {record_id!r} has no official execution answer"
                )
            normalized_table = item.get("table", {})
            raw_table = normalized_table.get("source_rows")
            if raw_table is None:
                if require_current_provenance:
                    raise ValueError(
                        f"Re-normalize record {record_id!r} to retain its complete source table"
                    )
                raw_table = [
                    [str(cell.get("raw", "")) for cell in row]
                    for row in normalized_table.get("rows", [])
                ]
            records.append(
                EvaluationRecord(
                    record_id=record_id,
                    question=doc["question"],
                    company_identifier=company,
                    gold_answer=gold_answer,
                    gold_display_answer=doc.get("answer_text"),
                    gold_program=reasoning.get("program"),
                    gold_inds=list(gold_indices),
                    gold_evidence=[str(text or "") for text in gold_indices.values()],
                    split=item.get("split", "dev"),
                    table=raw_table,
                )
            )

    id_set_digest(ids)
    if limit is not None and len(records) != limit:
        raise ValueError(f"Expected {limit} records, found {len(records)}")
    if len({record.split for record in records}) != 1:
        raise ValueError("A benchmark run must select exactly one dataset split")
    return records


def generate_markdown_report(summary: dict[str, Any]) -> str:
    taxonomy_table = "\n".join(
        f"| {cat} | {count} | {count / summary['total_samples']:.1%} |"
        for cat, count in summary["taxonomy_breakdown"].items()
    )

    return f"""# Autonomous Financial Reasoning Agent - Benchmark Report

**Dataset Split:** {summary['split']}

**Total Samples:** {summary['total_samples']}

**Scope:** {summary['scope']}

**Displayed Execution Accuracy:** {summary['execution_accuracy']:.2%}

**FinQA Program Execution Accuracy (Acc_exe):** {summary['program_execution_accuracy']:.2%}

**Program Structure Match Rate:** {summary['program_accuracy']:.2%}

## Information Retrieval (IR) Multi-Hop Metrics
* **Recall@5:** {summary['mean_recall_at_5']:.2%}
* **Precision@5:** {summary['mean_precision_at_5']:.2%}
* **Mean Reciprocal Rank (MRR):** {summary['mean_mrr']:.4f}
* **NDCG@5:** {summary['mean_ndcg_at_5']:.4f}

## Latency & Performance
* **Mean Wall Latency:** {summary['mean_latency_ms']:.2f} ms
* **P95 Latency:** {summary['p95_latency_ms']:.2f} ms
* **Observed model cost** {summary['known_total_cost_usd']:.6f} USD
* **Calls with unknown cost** {summary['unknown_cost_calls']}

## Causal Failure Taxonomy
| Category | Count | Proportion |
| :--- | :---: | :---: |
{taxonomy_table}
"""


def _write_json(path: Path, data: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a new FinQA candidate benchmark")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("data/processed/normalized/finqa_dev_normalized.jsonl"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--scope", choices=("known-record", "open-corpus"), required=True
    )
    parser.add_argument(
        "--corpus-id",
        required=True,
        help="Immutable retrieval index snapshot identifier",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--display-abs-tol", type=float, default=0.0)
    parser.add_argument("--display-rel-tol", type=float, default=0.0)
    parser.add_argument(
        "--save-states", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    from langgraph.checkpoint.memory import MemorySaver

    from agent.graph import create_financial_agent
    from agent.llm import model_for_task
    from knowledge.vector_store.search import embedding_model_name

    records = load_finqa_records(
        str(args.data_file), limit=args.limit, require_current_provenance=True
    )
    config = RunConfig(
        dataset_sha256=hash_file(args.data_file),
        record_ids_sha256=id_set_digest(rec.record_id for rec in records),
        split=records[0].split,
        scope=args.scope,
        corpus_id=args.corpus_id,
        display_abs_tol=args.display_abs_tol,
        display_rel_tol=args.display_rel_tol,
    )
    manifest = RunManifest(
        config=config,
        execution="candidate-runtime",
        code_sha256=code_digest(Path(__file__).resolve().parent.parent),
        models=tuple(
            [
                (task, model_for_task(task))
                for task in ("planner", "program", "verifier", "synthesis")
            ]
            + [("embedding", embedding_model_name())]
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(args.output_dir / "run_manifest.json", manifest.artifact())
    states_dir = args.output_dir / "states"
    if args.save_states:
        states_dir.mkdir()

    evaluator = FinQAEvaluator(
        agent_runner=create_financial_agent(checkpointer=MemorySaver()),
        scope=config.scope,
        display_abs_tol=config.display_abs_tol,
        display_rel_tol=config.display_rel_tol,
    )
    results: list[EvaluationResult] = []
    state_hashes: dict[str, str] = {}
    print(f"Executing candidate runtime on {len(records)} records")
    for index, record in enumerate(records, start=1):
        result, state = evaluator.run_instance(record)
        results.append(result)
        if args.save_states:
            state_path = states_dir / f"{safe_record_filename(record.record_id)}.json"
            _write_json(state_path, {"record_id": record.record_id, "state": state})
            state_hashes[record.record_id] = hash_file(state_path)
        if index % 10 == 0 or index == len(records):
            print(f"Processed {index}/{len(records)} instances")

    if hash_file(args.data_file) != config.dataset_sha256:
        raise ValueError("The dataset changed during this run")
    n = len(results)
    latencies = sorted(result.latency_ms for result in results)
    display = [
        result.display_correct
        for result in results
        if result.display_correct is not None
    ]
    summary = {
        "split": config.split,
        "scope": config.scope,
        "total_samples": n,
        "execution_accuracy": round(sum(r.execution_correct for r in results) / n, 4),
        "program_execution_accuracy": round(
            sum(r.program_execution_correct for r in results) / n, 4
        ),
        "program_accuracy": round(sum(r.program_correct for r in results) / n, 4),
        "display_accuracy": (
            round(sum(bool(x) for x in display) / len(display), 4) if display else None
        ),
        "mean_recall_at_5": round(
            sum(r.ir_metrics["recall_at_k"] for r in results) / n, 4
        ),
        "mean_precision_at_5": round(
            sum(r.ir_metrics["precision_at_k"] for r in results) / n, 4
        ),
        "mean_mrr": round(sum(r.ir_metrics["mrr"] for r in results) / n, 4),
        "mean_ndcg_at_5": round(sum(r.ir_metrics["ndcg_at_k"] for r in results) / n, 4),
        "mean_latency_ms": round(sum(latencies) / n, 2),
        "p95_latency_ms": round(latencies[min(int(n * 0.95), n - 1)], 2),
        "known_total_cost_usd": round(sum(r.observed_cost_usd for r in results), 6),
        "unknown_cost_calls": sum(r.unknown_cost_calls for r in results),
        "taxonomy_breakdown": dict(Counter(r.taxonomy.value for r in results)),
    }
    _write_json(args.output_dir / "benchmark_summary.json", summary)
    with (args.output_dir / "benchmark_report.md").open("x", encoding="utf-8") as out:
        out.write(generate_markdown_report(summary))
    serialized = [r.model_dump(mode="json") for r in results]
    _write_json(
        args.output_dir / "benchmark_results.json",
        {
            **manifest.artifact(),
            "split": config.split,
            "total_samples": n,
            "execution_accuracy": summary["execution_accuracy"],
            "results": serialized,
            "results_sha256": hash_json(serialized),
            "state_file_sha256": state_hashes,
        },
    )
    print(f"Displayed execution accuracy: {summary['execution_accuracy']:.2%}")
    print(
        f"FinQA program execution accuracy: {summary['program_execution_accuracy']:.2%}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
