"""Evaluation executor for scoring agent execution over FinQA benchmark records."""

from __future__ import annotations

import time
from typing import Any
from pydantic import BaseModel, Field

from agent.state.schema import AgentStateV1
from evaluation.metrics import compute_ir_metrics, is_numeric_match, is_program_match, parse_float_safe
from evaluation.taxonomy import FailureCategory, TaxonomyAttribution, classify_failure


class EvaluationRecord(BaseModel):
    record_id: str
    question: str
    company_identifier: str | None = None
    gold_answer: float | str
    gold_program: str | None = None
    gold_inds: list[str] = Field(default_factory=list)
    split: str = "dev"


class EvaluationResult(BaseModel):
    record_id: str
    predicted_answer: float | None
    gold_answer: float | None
    execution_correct: bool
    program_correct: bool
    ir_metrics: dict[str, float]
    taxonomy: FailureCategory
    taxonomy_explanation: str
    latency_ms: float
    token_cost_usd: float
    trace_id: str


class FinQAEvaluator:
    """Executes records against the LangGraph agent and computes benchmarks."""

    def __init__(self, agent_runner: Any | None = None) -> None:
        self.agent = agent_runner

    def evaluate_instance(
        self,
        record: EvaluationRecord,
        simulated_state: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        start_time = time.perf_counter()

        if simulated_state is not None:
            final_state = simulated_state
        elif self.agent is not None:
            initial_state = AgentStateV1(
                input=record.question,
                company_identifier=record.company_identifier,
            )
            config = {"configurable": {"thread_id": f"eval_{record.record_id}"}}
            final_state = self.agent.invoke(initial_state.model_dump(), config=config)
        else:
            raise ValueError("No agent runner or simulated state provided.")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 1. Extract Prediction — from BOTH sequential (tool_results) and
        # parallel (sub_task_results) execution paths (Phase 7 consistency)
        tool_results = list(final_state.get("tool_results", []))
        seen = {
            (r.get("task_id"), r.get("tool_name"))
            for r in tool_results
            if r.get("task_id")
        }
        for tid, env in (final_state.get("sub_task_results") or {}).items():
            if isinstance(env, dict) and (tid, env.get("tool_name")) not in seen:
                tool_results.append(env)
        math_results = [
            r.get("data", {}).get("result")
            for r in tool_results
            if r.get("tool_name") in {"safe_math", "code_interpreter"} and r.get("data")
        ]
        raw_pred = math_results[-1] if math_results else final_state.get("final_answer")
        pred_float = parse_float_safe(raw_pred)
        gold_float = parse_float_safe(record.gold_answer)

        # 2. Compute Metric Accuracies
        exe_match = is_numeric_match(pred_float, gold_float)

        executed_exprs = [
            str(r.get("data", {}).get("expression", ""))
            for r in tool_results
            if r.get("tool_name") == "safe_math"
        ]
        prog_match = (
            any(is_program_match(e, record.gold_program) for e in executed_exprs)
            if record.gold_program
            else False
        )

        # 3. Compute IR Metrics
        retrieved_ids: list[str] = []
        for r in tool_results:
            data = r.get("data")
            if isinstance(data, dict):
                for rec in data.get("records", []):
                    retrieved_ids.append(str(rec.get("row_label", "")))
                for ch in data.get("chunks", []):
                    retrieved_ids.append(f"{ch.get('section')}_{ch.get('chunk_index')}")

        ir_scores = compute_ir_metrics(retrieved_ids, record.gold_inds, k=5)

        # 4. Attribute Causal Failure
        attr: TaxonomyAttribution = classify_failure(
            state=final_state,
            gold_answer=record.gold_answer,
            gold_program=record.gold_program,
            gold_inds=record.gold_inds,
            recall_at_k=ir_scores["recall_at_k"],
        )

        return EvaluationResult(
            record_id=record.record_id,
            predicted_answer=pred_float,
            gold_answer=gold_float,
            execution_correct=exe_match,
            program_correct=prog_match,
            ir_metrics=ir_scores,
            taxonomy=attr.category,
            taxonomy_explanation=attr.explanation,
            latency_ms=round(elapsed_ms, 3),
            token_cost_usd=0.0,
            trace_id=final_state.get("trace_id", "trace_simulated"),
        )