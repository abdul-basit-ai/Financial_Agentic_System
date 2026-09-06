"""Comprehensive validation suite for Phase 4 Agent Tools."""

from __future__ import annotations

import pytest

from agent.tools.decomposer import decompose_query, DecompositionInput
from agent.tools.fusion_tool import (
    ContextFusionInput,
    reciprocal_rank_fusion,
)
from agent.tools.graph_tool import (
    FORBIDDEN_CYPHER_MUTATIONS,
    GraphMetricRecord,
    GraphQueryInput,
)
from agent.tools.safe_math import (
    ASTSecurityError,
    SafeMathEvaluator,
    SafeMathInput,
    safe_math_tool,
)
from agent.tools.vector_tool import VectorChunkRecord, VectorSearchInput


# =====================================================================
# Safe Math Tool Tests
# =====================================================================


def test_safe_math_standard_arithmetic() -> None:
    evaluator = SafeMathEvaluator()
    assert evaluator.evaluate("100 + 250") == 350.0
    assert evaluator.evaluate("(100 - 20) / 4") == 20.0
    assert evaluator.evaluate("2 ** 3") == 8.0
    assert evaluator.evaluate("-150.5 + 50") == -100.5


def test_safe_math_financial_symbols_and_percentages() -> None:
    evaluator = SafeMathEvaluator()
    # Currency symbols stripped
    assert evaluator.evaluate("$150.50 + $49.50") == 200.0
    assert evaluator.evaluate("€1,200 - €200") == 1000.0
    # Percentage notation conversion
    assert evaluator.evaluate("15% * 200") == 30.0
    assert evaluator.evaluate("100 + 10%") == 100.1


def test_safe_math_finqa_dsl_functions() -> None:
    evaluator = SafeMathEvaluator()
    assert evaluator.evaluate("add(100, 200)") == 300.0
    assert evaluator.evaluate("subtract(150, 50)") == 100.0
    assert evaluator.evaluate("divide(100, 4)") == 25.0
    assert evaluator.evaluate("multiply(10, 5)") == 50.0
    assert evaluator.evaluate("exp(2, 3)") == 8.0
    assert evaluator.evaluate("greater(10, 5)") == 1.0
    assert evaluator.evaluate("greater(2, 5)") == 0.0

    # FinQA table aggregations
    assert evaluator.evaluate("table_sum(10, 20, 30)") == 60.0
    assert evaluator.evaluate("table_max(5, 25, 12)") == 25.0
    assert evaluator.evaluate("table_min(5, 25, 12)") == 5.0
    assert evaluator.evaluate("table_average(10, 20, 30)") == 20.0


def test_safe_math_handles_division_by_zero() -> None:
    evaluator = SafeMathEvaluator()
    with pytest.raises(ZeroDivisionError):
        evaluator.evaluate("100 / 0")
    with pytest.raises(ZeroDivisionError):
        evaluator.evaluate("divide(100, 0)")


def test_safe_math_blocks_malicious_code_injection() -> None:
    evaluator = SafeMathEvaluator()
    with pytest.raises(ASTSecurityError):
        evaluator.evaluate("__import__('os').system('ls')")
    with pytest.raises(ASTSecurityError):
        evaluator.evaluate("globals()")
    with pytest.raises(ASTSecurityError):
        evaluator.evaluate("(lambda x: x + 1)(2)")


def test_safe_math_instrumented_envelope() -> None:
    result = safe_math_tool(SafeMathInput(expression="subtract(500, 200)"))
    assert result.success is True
    assert result.data is not None
    assert result.data.result == 300.0
    assert result.metrics.latency_ms >= 0.0


# =====================================================================
# Context Fusion Tool Tests (RRF Math & None Guards)
# =====================================================================


def test_reciprocal_rank_fusion_with_none_values() -> None:
    graph_data = [
        GraphMetricRecord(
            company="Apple",
            report_id="r1",
            row_label="Revenue",
            category="revenue",
            year=2020,
            amount=1000.0,
            normalized_amount=None,  # Tests None guard
        ),
        GraphMetricRecord(
            company="Apple",
            report_id="r1",
            row_label="Footnote Item",
            category="other",
            year=None,
            amount=None,  # Tests None guard
            normalized_amount=None,
        ),
    ]

    vector_data = [
        VectorChunkRecord(
            record_id="r1",
            filename="aapl.txt",
            section="pre_text",
            chunk_index=0,
            text_content="Strong revenue driven by iPhone sales.",
            similarity_score=0.92,
        )
    ]

    output = reciprocal_rank_fusion(
        graph_records=graph_data,
        vector_chunks=vector_data,
        graph_weight=1.0,
        vector_weight=1.0,
        k=60,
    )

    assert output.total_input_items == 3
    assert output.retained_items == 3
    assert "N/A" in output.formatted_context
    assert output.fused_items[0].rrf_score == pytest.approx(1.0 / 61.0, rel=1e-3)


# =====================================================================
# Query Decomposition Tool Tests
# =====================================================================


def test_query_decomposer_identifies_yoy_and_metrics() -> None:
    q = "What was the percentage change in Apple's revenue between 2019 and 2020?"
    out = decompose_query(q, company="AAPL")

    assert out.is_multi_hop is True
    assert len(out.sub_tasks) == 3
    assert out.sub_tasks[0].query_payload["year"] == 2019
    assert out.sub_tasks[0].query_payload["metric_name"] == "revenue"
    assert out.sub_tasks[1].query_payload["year"] == 2020
    assert out.sub_tasks[1].query_payload["metric_name"] == "revenue"
    assert out.sub_tasks[2].target_tool == "safe_math"


# =====================================================================
# Graph Security & Mutation Guard Tests
# =====================================================================


def test_graph_tool_blocks_cypher_mutations() -> None:
    forbidden_queries = [
        "MATCH (n) DETACH DELETE n",
        "MATCH (c:Company) SET c.name = 'Hacked'",
        "CREATE (n:BadNode {id: 1})",
        "MERGE (c:Company {id: 'hack'})",
        "DROP CONSTRAINT company_id_unique",
    ]
    for q in forbidden_queries:
        assert bool(FORBIDDEN_CYPHER_MUTATIONS.search(q)) is True

    safe_query = "MATCH (c:Company)-[:FILED]->(r:Report) RETURN c.name, r.year"
    assert bool(FORBIDDEN_CYPHER_MUTATIONS.search(safe_query)) is False


# =====================================================================
# JSON Schema Export Tests (MCP Phase 10 Readiness)
# =====================================================================


def test_pydantic_json_schema_exportability() -> None:
    schemas = [
        SafeMathInput.model_json_schema(),
        GraphQueryInput.model_json_schema(),
        VectorSearchInput.model_json_schema(),
        ContextFusionInput.model_json_schema(),
        DecompositionInput.model_json_schema(),
    ]
    for s in schemas:
        assert "type" in s
        assert "properties" in s
        assert isinstance(s["properties"], dict)