"""Causal failure attribution taxonomy for multi-hop financial reasoning."""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel

from evaluation.metrics import is_numeric_match, is_program_match


class FailureCategory(str, Enum):
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
) -> TaxonomyAttribution:
    """Classifies root cause failure using sequential causal elimination."""
    final_answer = state.get("final_answer")
    tool_results = state.get("tool_results", [])
    hitl_status = state.get("hitl_status", "NONE")
    scratchpad = state.get("scratchpad", [])

    # 1. Check Guardrail Interrupts & Aborts
    if hitl_status == "REJECTED":
        return TaxonomyAttribution(
            category=FailureCategory.GUARDRAIL_INTERRUPT,
            explanation="Execution was terminated by human review or compliance guardrail rejection.",
            diagnostics={"hitl_status": hitl_status},
        )

    # 2. Check Timeouts or Unhandled Crashes
    has_crash = any(not r.get("success", True) for r in tool_results)
    if not final_answer and has_crash:
        return TaxonomyAttribution(
            category=FailureCategory.TIMEOUT_OR_CRASH,
            explanation="Workflow failed to synthesize an answer due to tool-level exceptions or timeouts.",
            diagnostics={"tool_results": tool_results},
        )

    # 3. Check Numerical Correctness
    # Merge sequential and parallel execution envelopes (Phase 7 consistency)
    tool_results = list(tool_results)
    seen = {
        (r.get("task_id"), r.get("tool_name"))
        for r in tool_results
        if r.get("task_id")
    }
    for tid, env in (state.get("sub_task_results") or {}).items():
        if isinstance(env, dict) and (tid, env.get("tool_name")) not in seen:
            tool_results.append(env)

    math_results = [
        r.get("data", {}).get("result")
        for r in tool_results
        if r.get("tool_name") in {"safe_math", "code_interpreter"} and r.get("data")
    ]
    candidate_answer = math_results[-1] if math_results else final_answer

    is_exe_correct = is_numeric_match(candidate_answer, gold_answer) or is_numeric_match(
        final_answer, gold_answer
    )

    if is_exe_correct:
        return TaxonomyAttribution(
            category=FailureCategory.CORRECT,
            explanation="Output verified within allowable numerical tolerance bounds.",
            diagnostics={"predicted": candidate_answer, "gold": gold_answer},
        )

    # 4. Causal Step 1: Retrieval Miss
    if gold_inds and recall_at_k < 1.0:
        return TaxonomyAttribution(
            category=FailureCategory.RETRIEVAL_MISS,
            explanation=f"Information deficit: recall@k ({recall_at_k:.2f}) failed to retrieve all gold evidence items.",
            diagnostics={"recall_at_k": recall_at_k, "gold_inds": gold_inds},
        )

    # 5. Causal Step 2: Planning Error
    executed_expressions = [
        str(r.get("data", {}).get("expression", ""))
        for r in tool_results
        if r.get("tool_name") == "safe_math"
    ]
    matched_prog = (
        any(is_program_match(expr, gold_program) for expr in executed_expressions)
        if gold_program
        else False
    )

    if gold_program and not matched_prog:
        return TaxonomyAttribution(
            category=FailureCategory.PLANNING_ERROR,
            explanation="Planner constructed a mathematically non-isomorphic DSL expression relative to gold program.",
            diagnostics={"executed_expressions": executed_expressions, "gold_program": gold_program},
        )

    # 6. Causal Step 3: Arithmetic Execution Failure
    if any(r.get("tool_name") == "safe_math" and not r.get("success") for r in tool_results):
        return TaxonomyAttribution(
            category=FailureCategory.ARITHMETIC_ERROR,
            explanation="Safe math AST parser failed on division by zero, float overflow, or syntax error.",
            diagnostics={"tool_results": tool_results},
        )

    # 7. Causal Step 4: Synthesis Hallucination
    return TaxonomyAttribution(
        category=FailureCategory.SYNTHESIS_HALLUCINATION,
        explanation="Intermediate calculations matched, but final synthesis output omitted or misstated the result.",
        diagnostics={"final_answer": final_answer, "gold_answer": gold_answer},
    )