"""Regression tests for the audit fixes: loader/query ID contract,
content-based IR metrics, regression-runner contract, and LLM integration
fallbacks. Complements the per-phase suites; everything here runs offline."""

from __future__ import annotations

import json

import pytest

from agent.llm import extract_numerals
from agent.nodes.plan_node import LLMPlan, _extract_json, _validate_llm_plan
from agent.nodes.synthesize_node import _is_grounded
from agent.tools.graph_tool import _slugify
from evaluation.evaluator import EvaluationRecord, FinQAEvaluator
from evaluation.metrics import compute_ir_metrics_content, evidence_hit
from ingestion.entity_extractor import _extract_company_identifier

# =====================================================================
# A1: Graph loader <-> retrieval tool ID contract
# =====================================================================


def test_graph_tool_slugify_matches_loader_convention() -> None:
    """metric/company IDs built at query time must be byte-identical to the
    IDs written by ingestion.graph_loader at load time."""
    from ingestion.graph_loader import slugify as loader_slugify

    samples = [
        "operating income",
        "operating_income",
        "Net Income",
        "Amazon Inc",
        "stockholders' equity",
        "  diluted  EPS  ",
        "revenue",
    ]
    for sample in samples:
        assert _slugify(sample) == loader_slugify(sample), sample


def test_slugify_output_shape() -> None:
    assert _slugify("operating income") == "operating_income"
    assert _slugify("Stockholders' Equity") == "stockholders_equity"
    assert _slugify("   ") == "unknown"


def test_single_letter_company_codes_are_preserved() -> None:
    """FinQA filename prefixes are company codes; single letters are real
    tickers (V = Visa, C = Citigroup) and must not be dropped."""
    assert _extract_company_identifier("V/2008/page_17.pdf") == "V"
    assert _extract_company_identifier("C/2017/page_328.pdf") == "C"
    assert _extract_company_identifier("AMZN/2019/page_33.pdf") == "AMZN"
    assert _extract_company_identifier("") == ""
    assert _extract_company_identifier("PAGE/2001/x.pdf") == ""  # blacklisted


# =====================================================================
# A2: Content-based IR metrics over real FinQA gold_inds format
# =====================================================================


GOLD_TABLE_EVIDENCE = (
    "company the american express of payments volume ( billions ) is 637 ; "
    "the american express of total volume ( billions ) is 647 ;"
)
GOLD_TEXT_EVIDENCE = "oil gas ngls total ( mmbbls ) ( bcf ) ( mmbbls ) ( mmboe ) ."


def test_evidence_hit_via_number_and_label() -> None:
    retrieved_row = "payments volume 637.0 637000000000.0 2017"
    assert evidence_hit(GOLD_TABLE_EVIDENCE, retrieved_row)


def test_evidence_miss_without_shared_number() -> None:
    assert not evidence_hit(GOLD_TABLE_EVIDENCE, "total revenues 42.0 42000000.0 2017")


def test_evidence_hit_pure_narrative_gold() -> None:
    chunk = "Production volumes: oil, gas, NGLs and totalboe for the fiscal year."
    assert evidence_hit(GOLD_TEXT_EVIDENCE, chunk)
    assert not evidence_hit(GOLD_TEXT_EVIDENCE, "Interest expense declined in 2019.")


def test_content_ir_metrics_recall_precision_ranking() -> None:
    retrieved = [
        "payments volume 637.0 637000000000.0",  # covers gold item 1
        "unrelated risk factor text about foreign exchange",  # irrelevant
        "total volume 647.0 647000000000.0",  # covers gold item 2
    ]
    scores = compute_ir_metrics_content(retrieved, [GOLD_TABLE_EVIDENCE], k=5)
    assert scores["recall_at_k"] == 1.0
    assert scores["precision_at_k"] == pytest.approx(2 / 3, abs=1e-3)
    assert scores["mrr"] == 1.0

    # Rank-based: relevant at position 1 -> MRR 1.0; all-relevant NDCG = 1.0
    assert scores["ndcg_at_k"] == 1.0


def test_content_ir_metrics_zero_when_nothing_matches() -> None:
    scores = compute_ir_metrics_content(
        ["completely unrelated narrative"], [GOLD_TABLE_EVIDENCE], k=5
    )
    assert scores["recall_at_k"] == 0.0
    assert scores["precision_at_k"] == 0.0
    assert scores["mrr"] == 0.0


def test_content_ir_metrics_empty_gold_is_neutral() -> None:
    scores = compute_ir_metrics_content(["anything"], [], k=5)
    assert scores == {
        "recall_at_k": 1.0,
        "precision_at_k": 1.0,
        "mrr": 1.0,
        "ndcg_at_k": 1.0,
    }


def test_evaluator_uses_content_ir() -> None:
    """End-to-end through the evaluator: simulated state with a retrieved row
    covering the gold evidence must score recall 1.0 (the old ID-space metric
    could never score above 0 on real data)."""
    record = EvaluationRecord(
        record_id="rec_test_1",
        question="What were payments volume figures?",
        gold_answer="637",
        gold_inds=["table_3"],
        gold_evidence=[GOLD_TABLE_EVIDENCE],
    )
    simulated_state = {
        "tool_results": [
            {
                "task_id": "task_1",
                "tool_name": "graph_retrieval",
                "success": True,
                "data": {
                    "records": [
                        {
                            "company": "American Express",
                            "report_id": "report::x",
                            "row_label": "payments volume",
                            "category": "other",
                            "year": 2017,
                            "amount": 637.0,
                            "normalized_amount": 637000000000.0,
                        }
                    ],
                    "total_found": 1,
                },
                "error": None,
            }
        ],
        "sub_task_results": {},
        "final_answer": None,
        "trace_id": "trace_test",
    }
    result = FinQAEvaluator().evaluate_instance(record, simulated_state=simulated_state)
    assert result.ir_metrics["recall_at_k"] == 1.0
    assert result.ir_metrics["mrr"] == 1.0


# =====================================================================
# C1/C2: LLM planner parsing, validation, and offline fallback
# =====================================================================


def test_extract_json_plain_and_fenced() -> None:
    fenced = '```json\n{"a": 1}\n```'
    assert _extract_json(fenced) == {"a": 1}
    assert _extract_json('{"a": 1}') == {"a": 1}
    assert _extract_json('Sure! {"a": 1} hope that helps') == {"a": 1}
    assert _extract_json("no json here") is None


def test_validate_llm_plan_accepts_valid_plan() -> None:
    plan = LLMPlan(
        is_multi_hop=True,
        reasoning_plan="YoY delta",
        sub_tasks=[
            {
                "task_id": "task_1",
                "target_tool": "graph_retrieval",
                "query_payload": {"company_identifier": "AMZN", "year": 2019},
                "dependencies": [],
            },
            {
                "task_id": "task_2",
                "target_tool": "safe_math",
                "query_payload": {"expression": "task_1.amount * 2"},
                "dependencies": ["task_1"],
            },
        ],
    )
    assert _validate_llm_plan(plan) == []


def test_validate_llm_plan_rejects_forward_dependency_and_bare_math() -> None:
    plan = LLMPlan(
        sub_tasks=[
            {
                "task_id": "task_1",
                "target_tool": "safe_math",
                "query_payload": {"expression": "1 + 1"},
                "dependencies": ["task_2"],
            }
        ]
    )
    problems = _validate_llm_plan(plan)
    assert any("unknown/forward" in p for p in problems)
    assert any("without any retrieval" in p for p in problems)


def _plan_node_module():
    """Returns the plan_node MODULE for patching.

    agent/nodes/__init__ exports the plan_node FUNCTION under the same name,
    which shadows `import agent.nodes.plan_node as ...` attribute binding;
    importlib bypasses the shadow via sys.modules.
    """
    import importlib

    return importlib.import_module("agent.nodes.plan_node")


def test_plan_node_falls_back_to_rules_without_llm(monkeypatch) -> None:
    """No API key configured -> plan_node must produce the deterministic
    decomposition (this is what keeps CI and evaluation runnable offline)."""
    plan_module = _plan_node_module()

    monkeypatch.setattr(plan_module, "is_llm_enabled", lambda: False)

    state = _minimal_state(
        "What was the percentage change in Amazon operating revenue from 2019 to 2020?"
    )
    outcome = plan_module.plan_node(state)
    assert outcome["tool_calls"], "fallback plan must schedule tasks"
    tools = {c["target_tool"] for c in outcome["tool_calls"]}
    assert "graph_retrieval" in tools


def test_plan_node_uses_valid_llm_plan(monkeypatch) -> None:
    plan_module = _plan_node_module()

    monkeypatch.setattr(plan_module, "is_llm_enabled", lambda: True)

    llm_plan = {
        "is_multi_hop": True,
        "reasoning_plan": "Two-year delta with narrative backup.",
        "sub_tasks": [
            {
                "task_id": "task_1",
                "target_tool": "graph_retrieval",
                "query_payload": {
                    "company_identifier": "AMZN",
                    "year": 2019,
                    "metric_name": "operating revenue",
                },
                "dependencies": [],
            },
            {
                "task_id": "task_2",
                "target_tool": "graph_retrieval",
                "query_payload": {
                    "company_identifier": "AMZN",
                    "year": 2020,
                    "metric_name": "operating revenue",
                },
                "dependencies": [],
            },
            {
                "task_id": "task_3",
                "target_tool": "safe_math",
                "query_payload": {
                    "expression": "subtract(task_2.amount, task_1.amount)"
                },
                "dependencies": ["task_1", "task_2"],
            },
        ],
    }
    monkeypatch.setattr(
        plan_module,
        "invoke_llm",
        lambda **kwargs: type(
            "R",
            (),
            {
                "content": json.dumps(llm_plan),
                "model": "fake",
                "prompt_tokens": 120,
                "completion_tokens": 80,
                "cost_usd": 0.0001,
            },
        )(),
    )

    state = _minimal_state("Amazon revenue change 2019 to 2020")
    outcome = plan_module.plan_node(state)
    assert len(outcome["tool_calls"]) == 3
    assert outcome["tool_calls"][2]["dependencies"] == ["task_1", "task_2"]
    assert "llm:fake" in outcome["scratchpad"][0]


def test_plan_node_rejects_invalid_llm_plan(monkeypatch) -> None:
    """Malformed LLM JSON must silently degrade to the rule-based planner."""
    plan_module = _plan_node_module()

    monkeypatch.setattr(plan_module, "is_llm_enabled", lambda: True)
    monkeypatch.setattr(
        plan_module,
        "invoke_llm",
        lambda **kwargs: type(
            "R", (), {"content": "not json at all", "model": "fake"}
        )(),
    )

    state = _minimal_state("Why did operating income decrease in 2020?")
    outcome = plan_module.plan_node(state)
    assert outcome["tool_calls"], "fallback must still schedule retrieval"
    assert "rules]" in outcome["scratchpad"][0]


# =====================================================================
# C3: Synthesizer grounding gate
# =====================================================================


def test_extract_numerals_strips_commas() -> None:
    assert extract_numerals("Revenue rose 1,200 to 3,456.78 (-45)") == {
        "1200",
        "3456.78",
        "-45",
    }


def test_grounding_check_accepts_cited_figures() -> None:
    corpus = [
        "What was AMZN revenue growth?",
        "AMZN 2019 Operating revenue 286449 286449000000.0",
        "AMZN 2020 Operating revenue 386064 386064000000.0",
        "divide(subtract(task_2.amount, task_1.amount), task_1.amount) = 0.3760",
    ]
    answer = (
        "• Operating revenue grew from 286449 to 386064 (AMZN FY2020).\n"
        "• Growth rate: 0.3760."
    )
    assert _is_grounded(answer, corpus)


def test_grounding_check_rejects_fabricated_figures() -> None:
    corpus = ["AMZN 2020 Operating revenue 386064 386064000000.0"]
    answer = "Revenue grew by 42.7 percent to reach 550000 million dollars."
    assert not _is_grounded(answer, corpus)


def test_grounding_check_ignores_list_ordinals() -> None:
    corpus = ["AMZN 2020 revenue 386064"]
    answer = "1. Revenue was 386064.\n2. Verified against the filing."
    assert _is_grounded(answer, corpus)


# =====================================================================
# A3: regression runner contract
# =====================================================================


def test_run_regression_fails_loudly_without_states(
    tmp_path, monkeypatch, capsys
) -> None:
    """No recorded fixtures and no --live flag -> exit code 1 with a message
    that tells the operator how to generate fixtures (never a silent live run)."""
    from evaluation import run_regression

    data_file = tmp_path / "records.jsonl"
    record = {
        "record_id": "rec_missing",
        "document": {"question": "q", "execution_answer": "42"},
        "reasoning": {"program": None, "gold_indices": {}},
        "entities": {"company_names": []},
        "split": "dev",
    }
    data_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    states_dir = tmp_path / "states"  # does not exist

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_regression.py",
            "--data-file",
            str(data_file),
            "--states-dir",
            str(states_dir),
        ],
    )
    exit_code = run_regression.main()
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "no recorded states directory" in out
    assert "runner.py" in out


def _minimal_state(question: str):
    """Builds a minimal AgentStateV1-compatible object for node tests."""
    from agent.state.schema import AgentStateV1

    return AgentStateV1(input=question)
