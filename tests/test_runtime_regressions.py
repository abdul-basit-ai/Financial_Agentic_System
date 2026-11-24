import importlib
import json

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent.graph import create_financial_agent
from agent.llm import LLMCallResult, model_for_task
from agent.memory.promoter import MemoryPromotionEngine
from agent.nodes.compute_node import compute_node
from agent.nodes.parallel_nodes import sub_task_worker
from agent.nodes.plan_node import LLMPlan, _validate_llm_plan, plan_node
from agent.nodes.synthesize_node import (
    _generate_and_execute_program,
    _program_candidate,
    _resolve_cell,
    synthesize_node,
)
from agent.state.schema import AgentStateV1
from agent.tools.fusion_tool import reciprocal_rank_fusion
from agent.tools.graph_tool import GraphMetricRecord
from agent.tools.safe_math import SafeMathEvaluator

synthesis = importlib.import_module("agent.nodes.synthesize_node")


def test_explicit_offline_mode_prevents_provider_configuration_loading(monkeypatch):
    provider = importlib.import_module("agent.llm")
    monkeypatch.setenv("FINAGENT_DISABLE_LLM", "1")

    def forbidden_load():
        raise AssertionError("Provider configuration must not be loaded offline")

    monkeypatch.setattr(provider, "_load_env_file", forbidden_load)
    assert provider.is_llm_enabled() is False
    assert provider._get_model() is None


def response(payload):
    return LLMCallResult(json.dumps(payload), "test", 12, 7, 0.002)


def table():
    return {
        "record_id": "ABC/2020/page_4.pdf-1",
        "page": 4,
        "rows": [
            {
                "row_index": 1,
                "row_label": "Revenue",
                "cells": [
                    {
                        "column_index": i,
                        "column_header": str(2019 + i),
                        "amount": value,
                        "scale_multiplier": 1,
                        "quantity_kind": "currency",
                        "currency": "USD",
                    }
                    for i, value in enumerate([8.0, 10.0])
                ],
            }
        ],
    }


def call(task_id, tool, payload, deps=()):
    return {
        "task_id": task_id,
        "target_tool": tool,
        "payload": payload,
        "status": "PENDING",
        "dependencies": list(deps),
    }


def test_programs_require_unique_references_and_compatible_units():
    t = table()
    assert _program_candidate("999999", t, "What was revenue?") is None
    assert _program_candidate('add(cell("Revenue", "2020"), 1000)', t, "Total?") is None
    assert _resolve_cell(t, "Revenue", "2018") is None
    assert _resolve_cell(t, "Revenue", "2020") == 10
    t["rows"].append(t["rows"][0].copy())
    assert _resolve_cell(t, "Revenue", "2020") is None
    t = table()
    t["rows"][0]["cells"][0]["scale_multiplier"] = 1000
    assert (
        _program_candidate(
            'subtract(cell("Revenue","2020"),cell("Revenue","2019"))', t, "Change?"
        )
        is None
    )


@pytest.mark.parametrize(
    "verdict,success",
    [(None, False), ({"selected_index": True}, False), ({"selected_index": 0}, True)],
)
def test_program_fail_closed_and_usage(monkeypatch, verdict, success):
    monkeypatch.setenv("PROGRAM_SELF_CONSISTENCY_SAMPLES", "2")
    samples = iter(
        [
            response({"program": "999999"}),
            response({"program": 'cell("Revenue","2020")'}),
        ]
    )
    monkeypatch.setattr(
        synthesis,
        "invoke_llm",
        lambda task, **kwargs: (
            next(samples)
            if task == "program"
            else response(verdict) if verdict is not None else None
        ),
    )
    state = AgentStateV1(input="What was revenue in 2020?")
    out = _generate_and_execute_program(state, {"data": table()})
    assert len(out["llm_usage"]) == 3
    assert out["tool_results"][0]["success"] is success
    if success:
        data = out["tool_results"][0]["data"]
        assert data["result"] == 10
        assert data["citations"][0]["page"] == 4
        assert data["citations"][0]["column_header"] == "2020"


def test_computed_values_follow_dependencies_and_reject_ambiguous_ties():
    base = {
        "row_label": "Revenue",
        "report_id": "filing",
        "page": 4,
        "amount": 10.0,
        "year": 2020,
        "scale_multiplier": 1,
        "quantity_kind": "currency",
        "currency": "USD",
    }

    def run(records):
        return compute_node(
            AgentStateV1(
                input="Revenue in 2020",
                tool_calls=[
                    call(
                        "task_1",
                        "graph_retrieval",
                        {"metric_name": "Revenue", "year": 2020},
                    ),
                    call(
                        "task_2",
                        "safe_math",
                        {"expression": "add(task_1.amount,task_1.amount)"},
                        ["task_1"],
                    ),
                    call(
                        "task_3",
                        "safe_math",
                        {"expression": "add(task_2.amount,task_1.amount)"},
                        ["task_2", "task_1"],
                    ),
                ],
                sub_task_results={
                    "task_1": {
                        "tool_name": "graph_retrieval",
                        "success": True,
                        "data": {"records": records},
                    }
                },
                risk_evaluated=True,
            )
        )

    out = run([base])
    assert [r["data"]["result"] for r in out["tool_results"]] == [20.0, 30.0]
    assert len(out["tool_results"][-1]["data"]["citations"]) == 1
    assert out["risk_evaluated"] is False
    assert all(
        not r["success"] for r in run([base, {**base, "amount": 11.0}])["tool_results"]
    )


def test_replanning_uses_fresh_task_ids_and_retrieval(monkeypatch):
    monkeypatch.setattr(
        importlib.import_module("agent.nodes.plan_node"),
        "is_llm_enabled",
        lambda: False,
    )
    state = AgentStateV1(
        input="ABC revenue",
        iteration_count=1,
        tool_results=[{"task_id": "task_1", "success": False}],
    )
    out = plan_node(state)
    assert {c["task_id"] for c in out["tool_calls"]}.isdisjoint({"task_1", "task_2"})
    assert out["tool_calls"][0]["payload"]["top_k"] > 5


def test_planner_and_worker_validate_actual_payload():
    tool = call("task_1", "vector_retrieval", {"query_text": "revenue", "top_k": 500})
    plan = LLMPlan.model_validate(
        {
            "sub_tasks": [
                {
                    "task_id": "task_1",
                    "target_tool": "vector_retrieval",
                    "query_payload": tool["payload"],
                }
            ]
        }
    )
    assert _validate_llm_plan(plan)
    assert sub_task_worker(tool)["tool_results"][0]["success"] is False


def test_safe_math_argument_separators_and_finite_intermediate_values():
    e = SafeMathEvaluator()
    assert e.evaluate("add(1,200)") == 201
    assert e.evaluate("add(1,200) + 1,200") == 1401
    assert e.evaluate("1,200 + 1") == 1201
    for expression in (
        "1e309",
        "divide(1,1e309)",
        "multiply(1e308,1e308)",
        "True",
        "add(1,2, extra=55)",
    ):
        with pytest.raises(ValueError):
            e.evaluate(expression)


def test_fusion_preserves_same_year_distinct_columns():
    base = {
        "company": "ABC",
        "report_id": "r",
        "row_label": "US",
        "year": 2020,
        "amount": 10,
        "normalized_amount": 10,
        "category": "volume",
    }
    rows = [
        GraphMetricRecord(**base, column_index=i, column_header=name)
        for i, name in enumerate(["Volume", "Transactions"])
    ]
    out = reciprocal_rank_fusion(rows, [])
    assert len(out.fused_items) == 2
    assert len({i.item_id for i in out.fused_items}) == 2


def test_narrative_rejects_semantic_fabrication_and_records_usage(monkeypatch):
    monkeypatch.setattr(synthesis, "is_llm_enabled", lambda: True)
    verdict = {"id": "p1", "quote": "The company declared bankruptcy."}
    monkeypatch.setattr(synthesis, "invoke_llm", lambda **kwargs: response(verdict))
    state = AgentStateV1(
        input="Did the company declare bankruptcy?",
        iteration_count=3,
        evidence=[
            {
                "item_id": "p1",
                "source": "vector",
                "metadata": {
                    "record_id": "r",
                    "page": 1,
                    "text_content": "The company reported revenue of 10.",
                },
            }
        ],
    )
    out = synthesize_node(state)
    assert out["answer_status"] == "insufficient"
    assert out["llm_usage"][0]["cost_usd"] == 0.002


def test_default_model_and_task_specific_overrides(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "general")
    monkeypatch.setenv("PLANNER_MODEL", "planner")
    monkeypatch.setenv("EXTRACTION_MODEL", "extract")
    assert model_for_task("general") == "general"
    assert model_for_task("planner") == "planner"
    assert model_for_task("extraction") == "extract"


def test_rejected_state_never_promoted():
    class Store:
        def save_episode(self, entry):
            raise AssertionError("Rejected state must not be promoted")

    state = AgentStateV1(
        input="revenue",
        final_answer="Rejected",
        hitl_status="REJECT",
        is_terminal=True,
        sub_task_results={"task_1": {}},
        citations=[{"page": 1}],
    )
    assert MemoryPromotionEngine(Store()).promote_session("s", state) is None


def test_risk_reviews_generated_calculation_before_answer(monkeypatch):
    from agent.guardrails.risk_engine import FinancialRiskEngine, RiskAssessmentResult
    from agent.tools.base import ToolMetrics, ToolResult
    from agent.tools.table_tool import TableExtractOutput

    workers = importlib.import_module("agent.nodes.parallel_nodes")
    t = table()
    record_id = t["record_id"]
    monkeypatch.setenv("PROGRAM_SELF_CONSISTENCY_SAMPLES", "1")
    monkeypatch.setattr(
        importlib.import_module("agent.nodes.plan_node"),
        "is_llm_enabled",
        lambda: False,
    )
    monkeypatch.setattr(synthesis, "is_llm_enabled", lambda: True)
    monkeypatch.setattr(
        synthesis,
        "invoke_llm",
        lambda task, **kwargs: (
            response({"program": 'cell("Revenue","2020")'})
            if task == "program"
            else response({"selected_index": 0})
        ),
    )
    monkeypatch.setattr(
        workers,
        "table_extract_tool",
        lambda payload: ToolResult(
            tool_name="table_extract",
            success=True,
            metrics=ToolMetrics(latency_ms=0),
            data=TableExtractOutput(
                **t, headers=["2019", "2020"], total_rows=1, total_values=2
            ),
        ),
    )
    seen = []

    def risky(state):
        seen.extend(
            [r for r in state["tool_results"] if r.get("tool_name") == "safe_math"]
        )
        return RiskAssessmentResult(
            requires_hitl=bool(seen),
            composite_risk_score=1,
            reasons=[],
            inspected_metrics_count=1,
        )

    monkeypatch.setattr(FinancialRiskEngine, "evaluate_state", risky)
    graph = create_financial_agent(MemorySaver())
    cfg = {"configurable": {"thread_id": "review-calculation"}}
    paused = graph.invoke(
        AgentStateV1(
            input="What was revenue in 2020?", record_id=record_id
        ).model_dump(),
        cfg,
    )
    assert seen and seen[-1]["data"]["result"] == 10
    assert paused["__interrupt__"] and not paused.get("final_answer")
    final = graph.invoke(
        Command(resume={"action": "APPROVE", "analyst_id": "test"}), cfg
    )
    assert final["answer_status"] == "grounded"
    assert final["final_answer"].endswith("Answer: 10.0")
    assert final["answer_task_id"] == "program_1"


def test_rule_plan_calculation_uses_cited_years_in_correct_order(monkeypatch):
    from agent.tools.base import ToolMetrics, ToolResult
    from agent.tools.graph_tool import GraphQueryOutput

    workers = importlib.import_module("agent.nodes.parallel_nodes")
    planner = importlib.import_module("agent.nodes.plan_node")
    monkeypatch.setattr(planner, "is_llm_enabled", lambda: False)
    monkeypatch.setattr(synthesis, "is_llm_enabled", lambda: False)

    def retrieve(payload):
        row = GraphMetricRecord(
            company="ABC",
            report_id="ABC/2020/page_4.pdf-1",
            row_label="Revenue",
            category="revenue",
            year=payload.year,
            column_header=str(payload.year),
            amount={2019: 8.0, 2020: 10.0}[payload.year],
            page=4,
            scale_multiplier=1,
            quantity_kind="currency",
            currency="USD",
            normalized_amount=None,
        )
        return ToolResult(
            tool_name="graph_retrieval",
            success=True,
            metrics=ToolMetrics(latency_ms=0),
            data=GraphQueryOutput(
                company_identifier="ABC", records=[row], total_found=1
            ),
        )

    monkeypatch.setattr(workers, "graph_retrieval_tool", retrieve)
    graph = create_financial_agent(MemorySaver())
    state = AgentStateV1(
        input="What was the percentage change in ABC revenue from 2019 to 2020?",
        company_identifier="ABC",
    )
    out = graph.invoke(state.model_dump(), {"configurable": {"thread_id": "two-years"}})
    assert out["answer_status"] == "grounded"
    assert out["final_answer"].endswith("Answer: 0.25")
    assert out["answer_task_id"] == "task_4"
    assert {c["column_header"] for c in out["citations"]} == {"2019", "2020"}
    assert {c["page"] for c in out["citations"]} == {4}


def test_program_rejects_missing_pages_units_and_wrong_year():
    t = table()
    ref = 'cell("Revenue","2019")'
    assert _program_candidate(ref, t, "What was revenue in 2020?") is None
    t["record_id"] = "unlocated"
    del t["page"]
    assert _program_candidate(ref, t, "What was revenue?") is None
    t["page"] = 4
    t["rows"][0]["cells"][0]["scale_multiplier"] = None
    assert _program_candidate(ref, t, "What was revenue?") is None


def test_model_planned_math_requires_verifier_approval(monkeypatch):
    compute = importlib.import_module("agent.nodes.compute_node")
    monkeypatch.setattr(compute, "invoke_llm", lambda **kwargs: None)
    planned = call("task_2", "safe_math", {"expression": "task_1.amount"}, ["task_1"])
    planned["verify_formula"] = True
    row = {
        "report_id": "filing",
        "page": 1,
        "row_label": "Revenue",
        "year": 2020,
        "amount": 10.0,
        "scale_multiplier": 1.0,
        "quantity_kind": "currency",
        "currency": "USD",
    }
    state = AgentStateV1(
        input="Revenue",
        tool_calls=[
            call("task_1", "graph_retrieval", {"metric_name": "revenue"}),
            planned,
        ],
        sub_task_results={
            "task_1": {
                "tool_name": "graph_retrieval",
                "success": True,
                "data": {"records": [row]},
            }
        },
    )
    out = compute_node(state)
    assert not out["tool_results"][0]["success"]
    assert out["llm_usage"][0]["status"] == "failed"


@pytest.mark.parametrize(
    "supported,expected",
    [(True, "grounded"), (False, "insufficient"), (None, "insufficient")],
)
def test_narrative_quote_requires_independent_verification(
    monkeypatch, supported, expected
):
    quote = "Revenue increased because sales volume grew."

    def provider(task, **kwargs):
        if task == "synthesis":
            return response({"id": "p1", "quote": quote})
        return response({"supported": supported}) if supported is not None else None

    monkeypatch.setattr(synthesis, "is_llm_enabled", lambda: True)
    monkeypatch.setattr(synthesis, "invoke_llm", provider)
    state = AgentStateV1(
        input="Why did revenue increase?",
        iteration_count=3,
        evidence=[
            {
                "item_id": "p1",
                "source": "vector",
                "metadata": {"record_id": "r", "page": 2, "text_content": quote},
            }
        ],
    )
    out = synthesize_node(state)
    assert out["answer_status"] == expected
    assert [entry["stage"] for entry in out["llm_usage"]] == ["synthesis", "verifier"]
    if supported:
        assert out["citations"][0]["page"] == 2
        assert quote in out["final_answer"]
