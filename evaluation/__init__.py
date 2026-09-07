"""Evaluation and benchmarking suite package export."""

from evaluation.evaluator import EvaluationRecord, EvaluationResult, FinQAEvaluator
from evaluation.gate import GateDecision, compute_bootstrap_ci, compute_mcnemar_test, evaluate_gate
from evaluation.metrics import (
    canonicalize_program,
    compute_ir_metrics,
    is_numeric_match,
    is_program_match,
    parse_float_safe,
)
from evaluation.taxonomy import FailureCategory, TaxonomyAttribution, classify_failure

__all__ = [
    "parse_float_safe",
    "is_numeric_match",
    "canonicalize_program",
    "is_program_match",
    "compute_ir_metrics",
    "FailureCategory",
    "TaxonomyAttribution",
    "classify_failure",
    "EvaluationRecord",
    "EvaluationResult",
    "FinQAEvaluator",
    "GateDecision",
    "compute_mcnemar_test",
    "compute_bootstrap_ci",
    "evaluate_gate",
]