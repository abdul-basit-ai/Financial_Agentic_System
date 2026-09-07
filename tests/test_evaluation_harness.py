"""Validation suite for Phase 11 Evaluation Harness & Benchmarking Suite.

Verifies:
1. Dynamic tolerance matching with currency symbols, absolute and relative bounds, and scale equivalence.
2. Commutative AST canonicalization and program equivalence (add, multiply, table_sum).
3. Multi-hop IR metrics: Recall@K, Precision@K, MRR, and NDCG@K.
4. Causal failure taxonomy classification across failure modes.
5. McNemar chi-square test with closed-form p-value and bootstrap confidence intervals.
"""

from __future__ import annotations

import pytest

from evaluation.evaluator import EvaluationRecord, FinQAEvaluator
from evaluation.gate import compute_bootstrap_ci, compute_mcnemar_p_value_1df, compute_mcnemar_test
from evaluation.metrics import (
    canonicalize_program,
    compute_ir_metrics,
    is_numeric_match,
    is_program_match,
    parse_float_safe,
)
from evaluation.taxonomy import FailureCategory, classify_failure


# =====================================================================
# 1. Metric Accuracy & Scale Tests
# =====================================================================


def test_parse_float_safe() -> None:
    assert parse_float_safe("$1,250.50") == 1250.50
    assert parse_float_safe("€ 45.2%") == 45.2
    assert parse_float_safe("-350") == -350.0
    assert parse_float_safe("invalid") is None


def test_is_numeric_match_tolerances() -> None:
    # Direct match within absolute tolerance
    assert is_numeric_match(12.345, 12.3456, abs_tol=1e-3) is True
    # Relative tolerance on large corporate figures
    assert is_numeric_match(1_000_000.0, 1_005_000.0, rel_tol=0.01) is True
    # Percentage vs ratio scale equivalence (0.155 vs 15.5%)
    assert is_numeric_match(0.155, 15.5) is True
    assert is_numeric_match(15.5, 0.155) is True
    # Discrepancy beyond tolerance
    assert is_numeric_match(100.0, 115.0, rel_tol=0.01) is False


# =====================================================================
# 2. Commutative AST Isomorphism Tests
# =====================================================================


def test_commutative_ast_isomorphism() -> None:
    # add is commutative
    prog_a = "add(10, 20)"
    prog_b = "add(20, 10)"
    assert is_program_match(prog_a, prog_b) is True

    # multiply is commutative; nested operations resolve correctly
    p1 = "multiply(add(1, 2), 5)"
    p2 = "multiply(5, add(2, 1))"
    assert is_program_match(p1, p2) is True

    # subtract is NOT commutative
    s1 = "subtract(100, 50)"
    s2 = "subtract(50, 100)"
    assert is_program_match(s1, s2) is False

    # divide is NOT commutative
    d1 = "divide(100, 2)"
    d2 = "divide(2, 100)"
    assert is_program_match(d1, d2) is False


# =====================================================================
# 3. Information Retrieval (IR) Metrics Tests
# =====================================================================


def test_compute_ir_metrics() -> None:
    gold = ["row_revenue", "text_driver_1"]
    retrieved = ["row_assets", "row_revenue", "row_equity", "text_driver_1", "text_extra"]

    metrics = compute_ir_metrics(retrieved, gold, k=5)

    assert metrics["recall_at_k"] == 1.0  # Both found in top 5
    assert metrics["precision_at_k"] == 2.0 / 5.0  # 2 hits out of 5
    assert metrics["mrr"] == 0.5  # First hit at position 2 (1/2)
    assert metrics["ndcg_at_k"] > 0.6


# =====================================================================
# 4. Causal Failure Taxonomy Tests
# =====================================================================


def test_classify_failure_taxonomy() -> None:
    # 1. Correct
    state_ok = {
        "final_answer": "150.0",
        "tool_results": [{"tool_name": "safe_math", "data": {"result": 150.0}, "success": True}],
    }
    attr_ok = classify_failure(state_ok, gold_answer=150.0)
    assert attr_ok.category == FailureCategory.CORRECT

    # 2. Retrieval Miss
    state_miss = {"final_answer": "50.0", "tool_results": []}
    attr_miss = classify_failure(
        state_miss,
        gold_answer=100.0,
        gold_inds=["row_rev"],
        recall_at_k=0.0,
    )
    assert attr_miss.category == FailureCategory.RETRIEVAL_MISS

    # 3. Planning Error
    state_plan = {
        "final_answer": "20.0",
        "tool_results": [
            {"tool_name": "safe_math", "data": {"expression": "add(10, 10)"}, "success": True}
        ],
    }
    attr_plan = classify_failure(
        state_plan,
        gold_answer=100.0,
        gold_program="multiply(10, 10)",
        gold_inds=["row_rev"],
        recall_at_k=1.0,
    )
    assert attr_plan.category == FailureCategory.PLANNING_ERROR


# =====================================================================
# 5. Statistical Significance & Gating Tests
# =====================================================================


def test_mcnemar_exact_p_value() -> None:
    # If chi2 == 3.841, p-value should be ~ 0.05
    p_val = compute_mcnemar_p_value_1df(3.84146)
    assert p_val == pytest.approx(0.05, rel=1e-2)


def test_mcnemar_test_paired_contingency() -> None:
    base = [True] * 80 + [False] * 20
    # Candidate improves on 15 cases baseline missed, but regresses on only 1
    cand = [True] * 79 + [False] * 1 + [True] * 15 + [False] * 5

    chi2, p_val = compute_mcnemar_test(base, cand)
    assert chi2 > 3.841
    assert p_val < 0.05


def test_bootstrap_confidence_interval() -> None:
    flags = [True] * 80 + [False] * 20
    ci_low, ci_high = compute_bootstrap_ci(flags, n_bootstraps=500, seed=42)

    assert 0.70 <= ci_low <= 0.80
    assert 0.80 <= ci_high <= 0.90
    assert ci_low < ci_high


# =====================================================================
# 6. End-to-End FinQAEvaluator Test
# =====================================================================


def test_finqa_evaluator_instance() -> None:
    evaluator = FinQAEvaluator()

    record = EvaluationRecord(
        record_id="rec_eval_001",
        question="What was the growth in revenue?",
        company_identifier="MSFT",
        gold_answer=25.0,
        gold_program="divide(subtract(125, 100), 100)",
        gold_inds=["row_revenue"],
    )

    simulated_state = {
        "trace_id": "eval_trace_01",
        "final_answer": "Revenue grew by 25.0%",
        "tool_results": [
            {
                "tool_name": "safe_math",
                "success": True,
                "data": {
                    "result": 25.0,
                    "expression": "divide(subtract(125, 100), 100)",
                },
            }
        ],
    }

    result = evaluator.evaluate_instance(record, simulated_state=simulated_state)

    assert result.execution_correct is True
    assert result.program_correct is True
    assert result.taxonomy == FailureCategory.CORRECT
    assert result.predicted_answer == 25.0