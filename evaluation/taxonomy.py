"""Causal failure attribution taxonomy for multi-hop financial reasoning."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from evaluation.metrics import is_numeric_match, is_program_match, parse_answer_value


class FailureCategory(StrEnum):
    CORRECT = "CORRECT"
    RETRIEVAL_MISS = "RETRIEVAL_MISS"
    EXTRACTION_CORRUPTION = "EXTRACTION_CORRUPTION"
    PLANNING_ERROR = "PLANNING_ERROR"
    ARITHMETIC_ERROR = "ARITHMETIC_ERROR"
    SYNTHESIS_HALLUCINATION = "SYNTHESIS_HALLUCINATION"
    GUARDRAIL_INTERRUPT = "GUARDRAIL_INTERRUPT"
    TIMEOUT_OR_CRASH = "TIMEOUT_OR_CRASH"


class TaxonomyAttribution(BaseModel):
    category: FailureCategory
    explanation: str
    diagnostics: dict[str, Any]


def classify_failure(
    state: dict[str, Any],
    gold_answer: Any,
    gold_program: str | None = None,
    gold_inds: list[str] | None = None,
    recall_at_k: float = 1.0,
    *,
    execution_correct: bool | None = None,
    program_correct: bool | None = None,
) -> TaxonomyAttribution:
    """Classifies root cause failure using sequential causal elimination."""
    final_answer = state.get("final_answer")
    tool_results = state.get("tool_results", [])
    hitl_status = state.get("hitl_status", "NONE")

    if hitl_status in {"REJECTED", "REJECT", "PENDING"}:
        return TaxonomyAttribution(
            category=FailureCategory.GUARDRAIL_INTERRUPT,
            explanation="Human review did not approve execution.",
            diagnostics={"hitl_status": hitl_status},
        )

    has_crash = any(not r.get("success", True) for r in tool_results)
    if not final_answer and has_crash:
        return TaxonomyAttribution(
            category=FailureCategory.TIMEOUT_OR_CRASH,
            explanation="Workflow failed to synthesize an answer due to tool-level exceptions or timeouts.",
            diagnostics={"tool_results": tool_results},
        )

    tool_results = list(tool_results)
    seen = {
        (r.get("task_id"), r.get("tool_name")) for r in tool_results if r.get("task_id")
    }
    for tid, env in (state.get("sub_task_results") or {}).items():
        if isinstance(env, dict) and (tid, env.get("tool_name")) not in seen:
            tool_results.append(env)

    final_answer_value = parse_answer_value(final_answer)
    if (
        execution_correct
        if execution_correct is not None
        else is_numeric_match(final_answer_value, gold_answer)
    ):
        return TaxonomyAttribution(
            category=FailureCategory.CORRECT,
            explanation="The displayed answer matches the expected execution value.",
            diagnostics={"predicted": final_answer_value, "gold": gold_answer},
        )

    if gold_inds and recall_at_k < 1.0:
        return TaxonomyAttribution(
            category=FailureCategory.RETRIEVAL_MISS,
            explanation=f"Information deficit: recall@k ({recall_at_k:.2f}) failed to retrieve all gold evidence items.",
            diagnostics={"recall_at_k": recall_at_k, "gold_inds": gold_inds},
        )

    executed_expressions = [
        str(
            (r.get("data") or {}).get("program")
            or (r.get("data") or {}).get("expression", "")
        )
        for r in tool_results
        if r.get("tool_name") == "safe_math"
    ]
    matched_prog = (
        program_correct
        if program_correct is not None
        else bool(
            gold_program
            and executed_expressions
            and is_program_match(executed_expressions[-1], gold_program)
        )
    )

    if gold_program and not matched_prog:
        return TaxonomyAttribution(
            category=FailureCategory.PLANNING_ERROR,
            explanation="Planner constructed a mathematically non-isomorphic DSL expression relative to gold program.",
            diagnostics={
                "executed_expressions": executed_expressions,
                "gold_program": gold_program,
            },
        )

    if any(
        r.get("tool_name") == "safe_math" and not r.get("success") for r in tool_results
    ):
        return TaxonomyAttribution(
            category=FailureCategory.ARITHMETIC_ERROR,
            explanation="Safe math AST parser failed on division by zero, float overflow, or syntax error.",
            diagnostics={"tool_results": tool_results},
        )

    return TaxonomyAttribution(
        category=FailureCategory.SYNTHESIS_HALLUCINATION,
        explanation="The displayed output omitted or misstated the expected answer.",
        diagnostics={"final_answer": final_answer, "gold_answer": gold_answer},
    )
