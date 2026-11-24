"""Evaluation executor for scoring agent execution over FinQA benchmark records."""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.state.schema import AgentStateV1
from evaluation.evidence import score_evidence
from evaluation.finqa_program import canonicalize_program, execute_program
from evaluation.metrics import (
    execution_value,
    is_display_match,
    is_numeric_match,
    is_program_match,
    parse_answer_value,
)
from evaluation.taxonomy import FailureCategory, classify_failure


class EvaluationRecord(BaseModel):
    record_id: str
    question: str
    company_identifier: str | None = None
    gold_answer: float | str
    gold_display_answer: str | None = None
    table: list[list[str]] = Field(default_factory=list)
    gold_program: str | None = None
    gold_inds: list[str] = Field(default_factory=list)

    gold_evidence: list[str] = Field(default_factory=list)
    split: str = "dev"


class EvaluationResult(BaseModel):
    record_id: str
    predicted_answer: float | str | None
    gold_answer: float | str | None
    execution_correct: bool
    program_correct: bool
    program_execution_correct: bool
    display_correct: bool | None
    ir_metrics: dict[str, float]
    taxonomy: FailureCategory
    taxonomy_explanation: str
    latency_ms: float
    token_cost_usd: float | None
    observed_cost_usd: float
    unknown_cost_calls: int
    trace_id: str


class FinQAEvaluator:
    """Executes records against the LangGraph agent and computes benchmarks."""

    def __init__(
        self,
        agent_runner: Any | None = None,
        *,
        scope: Literal["known-record", "open-corpus"] = "known-record",
        display_abs_tol: float = 0.0,
        display_rel_tol: float = 0.0,
    ) -> None:
        self.agent = agent_runner
        self.scope = scope
        self.display_abs_tol = display_abs_tol
        self.display_rel_tol = display_rel_tol

        import uuid

        self.run_id = uuid.uuid4().hex

    def thread_id_for(self, record: EvaluationRecord) -> str:
        return f"eval_{self.run_id}_{record.record_id}"

    def _invoke_with_auto_approval(
        self, initial_state: AgentStateV1, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Invokes the graph; auto-approves any HITL interrupt it raises.

        Evaluation is non-interactive: a risk-triggered pause is a MEASUREMENT
        condition, not a workflow stop. Auto-approving keeps the benchmark
        about retrieval/math quality; paused-forever threads would otherwise
        all score as failures. Bounded at 3 approvals (graph max iterations).
        """
        assert self.agent is not None, "agent runner required for live evaluation"
        final_state: dict[str, Any] = dict(
            self.agent.invoke(initial_state.model_dump(), config=config)
        )
        try:
            from langgraph.types import Command
        except ImportError:
            return final_state

        for _ in range(3):
            snapshot = self.agent.get_state(config)
            if not snapshot or not snapshot.next:
                break
            resume: Any = Command(
                resume={
                    "action": "APPROVE",
                    "analyst_id": "eval_auto",
                    "feedback": "Auto-approved during benchmark evaluation.",
                    "overrides": {},
                }
            )
            final_state = dict(self.agent.invoke(resume, config=config))
        return final_state

    def evaluate_instance(
        self,
        record: EvaluationRecord,
        simulated_state: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        result, _ = self.run_instance(record, simulated_state)
        return result

    def run_instance(
        self,
        record: EvaluationRecord,
        simulated_state: dict[str, Any] | None = None,
    ) -> tuple[EvaluationResult, dict[str, Any]]:
        start_time = time.perf_counter()
        if simulated_state is not None:
            final_state = simulated_state
            elapsed_ms = 0.0
        elif self.agent is not None:
            initial_state = AgentStateV1(
                input=record.question,
                company_identifier=record.company_identifier,
                record_id=record.record_id if self.scope == "known-record" else None,
            )
            config = {"configurable": {"thread_id": self.thread_id_for(record)}}
            final_state = self._invoke_with_auto_approval(initial_state, config)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        else:
            raise ValueError("No agent runner or simulated state provided")

        tools = list(final_state.get("tool_results", []))
        seen = {
            (r.get("task_id"), r.get("tool_name")) for r in tools if r.get("task_id")
        }
        tools.extend(
            env
            for tid, env in (final_state.get("sub_task_results") or {}).items()
            if isinstance(env, dict) and (tid, env.get("tool_name")) not in seen
        )

        status = final_state.get("answer_status")
        final_answer = (
            final_state.get("final_answer")
            if status in {None, "grounded"}
            and final_state.get("hitl_status") not in {"REJECTED", "REJECT", "PENDING"}
            else None
        )
        prediction = parse_answer_value(final_answer)
        execution_correct = is_numeric_match(prediction, record.gold_answer)

        math = [
            envelope
            for envelope in tools
            if envelope.get("tool_name") == "safe_math"
            and envelope.get("success", True)
            and isinstance(envelope.get("data"), dict)
        ]
        task_id = final_state.get("answer_task_id")
        selected: dict[str, Any]
        if task_id:
            selected = next(
                (
                    envelope["data"]
                    for envelope in reversed(math)
                    if envelope.get("task_id") == task_id
                ),
                {},
            )
        else:
            selected = math[-1]["data"] if math and status is None else {}
        symbolic = str(selected.get("program") or "")
        program = (
            symbolic
            if canonicalize_program(symbolic)
            else str(selected.get("expression") or "")
        )
        program_correct = bool(
            record.gold_program and is_program_match(program, record.gold_program)
        )
        program_execution_correct = is_numeric_match(
            execute_program(program, record.table), record.gold_answer
        )
        display_correct = (
            is_display_match(
                final_answer,
                record.gold_display_answer,
                abs_tol=self.display_abs_tol,
                rel_tol=self.display_rel_tol,
            )
            if record.gold_display_answer is not None
            else None
        )

        ir_scores = score_evidence(
            record.record_id, record.gold_inds, record.gold_evidence, tools
        )
        attribution = classify_failure(
            state=final_state,
            gold_answer=record.gold_answer,
            gold_program=record.gold_program,
            gold_inds=record.gold_inds,
            recall_at_k=ir_scores["recall_at_k"],
            execution_correct=execution_correct,
            program_correct=program_correct,
        )

        costs = [item.get("cost_usd") for item in final_state.get("llm_usage", [])]
        unknowns = costs.count(None)
        observed_cost = round(sum(float(cost) for cost in costs if cost is not None), 6)
        result = EvaluationResult(
            record_id=record.record_id,
            predicted_answer=prediction,
            gold_answer=execution_value(record.gold_answer),
            execution_correct=execution_correct,
            program_correct=program_correct,
            program_execution_correct=program_execution_correct,
            display_correct=display_correct,
            ir_metrics=ir_scores,
            taxonomy=attribution.category,
            taxonomy_explanation=attribution.explanation,
            latency_ms=round(elapsed_ms, 3),
            token_cost_usd=None if unknowns else observed_cost,
            observed_cost_usd=observed_cost,
            unknown_cost_calls=unknowns,
            trace_id=final_state.get("trace_id", "trace_simulated"),
        )
        return result, final_state
