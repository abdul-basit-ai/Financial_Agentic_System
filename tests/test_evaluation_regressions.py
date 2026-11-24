from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from agent.graph import create_financial_agent
from agent.nodes import parallel_nodes
from agent.tools.base import ToolMetrics, ToolResult
from agent.tools.graph_tool import GraphMetricRecord, GraphQueryInput, GraphQueryOutput
from evaluation.evaluator import EvaluationRecord, FinQAEvaluator
from evaluation.finqa_program import execute_program
from evaluation.gate import evaluate_gate
from evaluation.manifest import RunConfig, RunManifest, hash_json, id_set_digest
from evaluation.metrics import (
    evidence_hit,
    is_numeric_match,
    is_program_match,
    parse_answer_value,
)
from evaluation.run_regression import _score_replay
from evaluation.runner import safe_record_filename
from ingestion.graph_loader import build_graph_payload
from ingestion.normalizer import normalize_record
from ingestion.parser import parse_record


def test_strict_answer_scoring_and_accounting_negatives() -> None:
    assert is_numeric_match("3%", 0.03)
    assert not is_numeric_match("3%", 3.0)
    assert not is_numeric_match(300, 3)
    assert not is_numeric_match(100, 102)
    assert parse_answer_value("Answer: $(1,125.25)") == -1125.25
    assert parse_answer_value("Answer: (3%) [source]") == -0.03
    assert parse_answer_value("Answer: 1.5e-05") == 0.000015
    assert parse_answer_value("Revenue in 2021 was 45") is None
    assert parse_answer_value("Answer: cannot confirm 45") is None
    assert is_numeric_match(parse_answer_value("Answer: Yes"), "yes")


def test_multistep_program_references_and_comparisons() -> None:
    program = "subtract(15, 10), divide(#0, 10), multiply(const_100, #1)"
    assert execute_program(program, []) == 50
    assert is_program_match(program, "multiply(100, divide(subtract(15, 10), 10))")
    assert not is_program_match("divide(#1, 7)", "divide(#1, 7)")
    assert not is_program_match("f(3, 4)", "f(3, 4)")
    assert execute_program("greater(15%, 0.2)", []) == "no"
    assert (
        execute_program(
            "table_average(Net sales, none), multiply(#0, const_m1)",
            [["Net sales", "$ 150 (1)", "200"]],
        )
        == -175
    )
    assert execute_program("table_sum(2020, 963202)", [["2020", "10", "20"]]) == 30
    assert execute_program("divide(1, 0), add(2, 2)", []) is None


def test_candidate_metric_uses_displayed_answer_instead_of_correct_intermediate() -> (
    None
):
    record = EvaluationRecord(
        record_id="AAA/2020/page_7.pdf-1",
        question="What changed?",
        gold_answer=20,
        gold_display_answer="20",
        gold_program="subtract(100, 80)",
    )
    state = {
        "answer_status": "grounded",
        "final_answer": "Answer: 21",
        "answer_task_id": "selected",
        "tool_results": [
            {
                "task_id": "selected",
                "tool_name": "safe_math",
                "success": True,
                "data": {"result": 20, "expression": "subtract(100, 80)"},
            },
            {
                "task_id": "unused",
                "tool_name": "safe_math",
                "success": True,
                "data": {"result": 21, "expression": "add(1, 20)"},
            },
        ],
        "llm_usage": [
            {"status": "completed", "cost_usd": 0.01},
            {"status": "failed", "cost_usd": None},
        ],
    }
    result = FinQAEvaluator().evaluate_instance(record, simulated_state=state)
    assert result.predicted_answer == 21
    assert not result.execution_correct
    assert not result.display_correct
    assert result.program_execution_correct
    assert result.program_correct
    assert result.token_cost_usd is None
    assert result.observed_cost_usd == 0.01
    assert result.unknown_cost_calls == 1


def test_evidence_requires_matching_amounts_and_record_scoped_exact_ids() -> None:
    gold = "Revenue in 2020 was 10 million and in 2021 was 20 million."
    assert not evidence_hit(
        gold, "Revenue in 2020 was 90 million and in 2021 was 80 million."
    )
    assert not evidence_hit(
        gold, "Revenue in 2020 was 20 million and in 2021 was 10 million."
    )
    assert not evidence_hit("Loss was (10 million)", "Loss was 10 million")
    assert evidence_hit(
        gold, "Reported revenue in 2020 was 10 million and in 2021 was 20 million."
    )

    record = EvaluationRecord(
        record_id="AAA/2021/page_7.pdf-1",
        question="What was revenue?",
        gold_answer=20,
        gold_inds=["text_0"],
        gold_evidence=[gold],
    )
    chunk: dict[str, Any] = {
        "record_id": "AAA/2021/page_9.pdf-1",
        "gold_ids": ["text_0"],
        "text_content": gold,
    }
    state = {
        "tool_results": [{"tool_name": "vector_retrieval", "data": {"chunks": [chunk]}}]
    }
    evaluator = FinQAEvaluator()
    assert evaluator.evaluate_instance(record, state).ir_metrics["recall_at_k"] == 0
    chunk["record_id"] = record.record_id
    chunk["gold_ids"] = ["text_1"]
    assert evaluator.evaluate_instance(record, state).ir_metrics["recall_at_k"] == 0
    chunk["gold_ids"] = ["text_0"]
    assert evaluator.evaluate_instance(record, state).ir_metrics["recall_at_k"] == 1


def _artifact(
    path: Path,
    ids: list[str],
    *,
    scope: str = "known-record",
    digest_ids: list[str] | None = None,
) -> str:
    manifest = RunManifest.model_validate(
        {
            "config": {
                "dataset_sha256": "a" * 64,
                "record_ids_sha256": id_set_digest(
                    digest_ids if digest_ids is not None else ids
                ),
                "split": "dev",
                "scope": scope,
                "corpus_id": "fixture-snapshot",
            },
            "execution": "candidate-runtime",
            "code_sha256": "b" * 64,
        }
    )
    results = [{"record_id": item, "execution_correct": True} for item in ids]
    path.write_text(
        json.dumps(
            {
                **manifest.artifact(),
                "results": results,
                "total_samples": len(results),
                "results_sha256": hash_json(results),
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def test_gate_requires_identical_ids_and_compatible_manifests(tmp_path: Path) -> None:
    baseline = _artifact(tmp_path / "base.json", ["one", "two"])
    candidate_path = tmp_path / "candidate.json"
    assert evaluate_gate(baseline, _artifact(candidate_path, ["two", "one"])).passed
    with pytest.raises(ValueError, match="identical"):
        evaluate_gate(baseline, _artifact(candidate_path, ["one"]))
    with pytest.raises(ValueError, match="unique"):
        evaluate_gate(
            baseline,
            _artifact(candidate_path, ["one", "one"], digest_ids=["one", "two"]),
        )
    with pytest.raises(ValueError, match="incompatible"):
        evaluate_gate(
            baseline, _artifact(candidate_path, ["one", "two"], scope="open-corpus")
        )
    with pytest.raises(ValueError, match="versioned manifest"):
        candidate_path.write_text('{"results":[]}', encoding="utf-8")
        evaluate_gate(baseline, str(candidate_path))

    config = RunConfig(
        dataset_sha256="c" * 64,
        record_ids_sha256="d" * 64,
        split="dev",
        scope="known-record",
        corpus_id="snapshot",
    )
    with pytest.raises(ValueError, match="frozen"):
        config.corpus_id = "different-snapshot"


def test_scorer_replay_fails_with_incomplete_coverage(tmp_path: Path) -> None:
    source = tmp_path / "sample.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "record_id": record,
                    "split": "dev",
                    "document": {
                        "question": f"Question {record}?",
                        "execution_answer": 100,
                    },
                }
            )
            for record in ("one", "two")
        ),
        encoding="utf-8",
    )
    (tmp_path / f"{safe_record_filename('one')}.json").write_text(
        json.dumps(
            {
                "record_id": "one",
                "state": {"input": "Question one?", "final_answer": "Answer: 100"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="No unambiguous saved state for record two"):
        _score_replay(source, tmp_path, None, 0)
    with pytest.raises(FileNotFoundError):
        _score_replay(tmp_path / "absent.jsonl", tmp_path, None, 0)


@pytest.mark.parametrize("mode", ["deterministic", "mocked_llm", "ambiguous"])
def test_candidate_runtime_executes_current_graph(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    llm = import_module("agent.llm")
    plan = import_module("agent.nodes.plan_node")
    synthesis = import_module("agent.nodes.synthesize_node")
    compute = import_module("agent.nodes.compute_node")

    planner_tasks = [
        {
            "task_id": f"task_{index}",
            "target_tool": "graph_retrieval",
            "query_payload": {
                "company_identifier": "XYZ",
                "metric_name": "revenue",
                "year": year,
            },
            "dependencies": [],
        }
        for index, year in enumerate((2020, 2021), start=1)
    ]
    planner_tasks.append(
        {
            "task_id": "task_3",
            "target_tool": "safe_math",
            "query_payload": {"expression": "subtract(task_2.amount, task_1.amount)"},
            "dependencies": ["task_1", "task_2"],
        }
    )
    invoked = []

    def fixture_provider(*args: Any, **kwargs: Any) -> Any:
        task = kwargs.get("task")
        invoked.append(task)
        if mode != "mocked_llm":
            pytest.fail("The deterministic candidate attempted a model call")
        if task == "planner":
            response = {"sub_tasks": planner_tasks}
        elif task == "verifier":
            response = {"selected_index": 0, "corrected_expression": None}
        else:
            pytest.fail(f"Unexpected fixture model task {task}")
        return llm.LLMCallResult(json.dumps(response), "fixture-model", 10, 5, 0.01)

    for module in (llm, plan, synthesis):
        monkeypatch.setattr(module, "is_llm_enabled", lambda: mode == "mocked_llm")
    for module in (llm, plan, synthesis, compute):
        monkeypatch.setattr(module, "invoke_llm", fixture_provider)

    record_id = "XYZ/2021/page_7.pdf-1"
    question = "What was the change in revenue from 2020 to 2021?"
    raw_table = [
        ["Statement (in USD millions)", "2020", "2021"],
        ["Revenue", "120", "150"],
    ]
    normalized = normalize_record(
        parse_record(
            "dev",
            {
                "id": record_id,
                "filename": "XYZ/2021/page_7.pdf",
                "table": raw_table,
                "qa": {"question": question},
            },
        )
    )
    payload = build_graph_payload(normalized)
    row = payload["rows"][0]

    def fixture_graph(query: GraphQueryInput) -> ToolResult[GraphQueryOutput]:
        assert query.record_id == record_id
        records = [
            GraphMetricRecord.model_validate(
                {
                    **cell,
                    "company": "XYZ",
                    "report_id": record_id,
                    "source_file": payload["report"]["source_file"],
                    "table_id": payload["table"]["id"],
                    "page": payload["report"]["page"],
                    "row_index": row["source_row_index"],
                    "row_label": row["label"],
                    "row_evidence_id": row["evidence_id"],
                    "gold_id": row["gold_id"],
                    "category": row["category"],
                }
            )
            for cell in payload["values"]
            if query.year == cell["year"]
        ]
        if mode == "ambiguous" and query.year == 2021:
            records.append(
                records[0].model_copy(
                    update={
                        "amount": 151,
                        "evidence_id": f"{record_id}::additional::column",
                    },
                )
            )
        return ToolResult(
            tool_name="graph_retrieval",
            success=True,
            metrics=ToolMetrics(latency_ms=0.0),
            data=GraphQueryOutput(
                company_identifier="XYZ", records=records, total_found=len(records)
            ),
        )

    monkeypatch.setattr(parallel_nodes, "graph_retrieval_tool", fixture_graph)
    result, state = FinQAEvaluator(
        create_financial_agent(checkpointer=MemorySaver()),
    ).run_instance(
        EvaluationRecord(
            record_id=record_id,
            question=question,
            company_identifier="XYZ",
            gold_answer=30,
            gold_program="subtract(150, 120)",
            gold_inds=["table_1"],
            table=raw_table,
        )
    )

    assert state["is_terminal"]
    assert result.ir_metrics["recall_at_k"] == 1.0
    if mode == "ambiguous":
        assert state["answer_status"] == "insufficient"
        assert result.predicted_answer is None
        assert not result.execution_correct
    else:
        assert result.execution_correct
        assert result.program_correct
        assert result.program_execution_correct
        assert {source["column_index"] for source in state["citations"]} == {1, 2}
        assert {source["page"] for source in state["citations"]} == {7}
    assert invoked == (["planner", "verifier"] if mode == "mocked_llm" else [])
