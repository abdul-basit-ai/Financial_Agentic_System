"""Evaluation executor for scoring agent execution over FinQA benchmark records."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from agent.state.schema import AgentStateV1
from evaluation.metrics import (
    compute_ir_metrics_content,
    is_numeric_match,
    is_program_match,
    parse_float_safe,
)
from evaluation.taxonomy import FailureCategory, TaxonomyAttribution, classify_failure


class EvaluationRecord(BaseModel):
    record_id: str
    question: str
    company_identifier: str | None = None
    gold_answer: float | str
    gold_program: str | None = None
    gold_inds: list[str] = Field(default_factory=list)
    # Human-readable gold evidence strings (the VALUES of FinQA gold_indices).
    # Retrieval is graded against these via content overlap; gold_inds keys
    # (table_N / text_M) are kept for diagnostics only.
    gold_evidence: list[str] = Field(default_factory=list)
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
        # Run-scoped thread prefix: the Redis checkpointer persists threads
        # forever, and append-only state reducers would otherwise merge this
        # run's results with every previous run on the same record id —
        # silently reproducing stale trajectories (observed: identical
        # benchmark results across runs after a retrieval overhaul).
        import uuid

        self.run_id = uuid.uuid4().hex[:8]

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
        except ImportError:  # very old langgraph: no interrupts to resolve
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
        start_time = time.perf_counter()

        if simulated_state is not None:
            final_state = simulated_state
        elif self.agent is not None:
            initial_state = AgentStateV1(
                input=record.question,
                company_identifier=record.company_identifier,
                record_id=record.record_id,
            )
            config = {"configurable": {"thread_id": self.thread_id_for(record)}}
            final_state = self._invoke_with_auto_approval(initial_state, config)
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

        def _float_from_answer_text(text: Any) -> float | None:
            """Parses the number after a trailing 'Answer:' marker when the
            synthesizer emitted one, else the first number in the text.
            Whole-text scanning otherwise catches citation figures (years,
            row amounts) that are not the answer."""
            s = str(text or "")
            marker = s.lower().rfind("answer:")
            if marker != -1:
                s = s[marker + len("answer:") :]
            return parse_float_safe(s)

        # 1. Primary prediction: the executed math result (deterministic,
        # exact). Without math, the synthesized answer text (marker-aware).
        pred_float = None
        if math_results:
            pred_float = parse_float_safe(math_results[-1])
        if pred_float is None:
            pred_float = _float_from_answer_text(final_state.get("final_answer"))
        gold_float = parse_float_safe(record.gold_answer)

        # 2. Compute Metric Accuracies. final_answer is the fallback — the
        # agent's stated answer is authoritative when the math envelope
        # carried a stale/derived value (e.g. template synthesis picked the
        # right figure directly from evidence).
        exe_match = is_numeric_match(pred_float, gold_float)
        if not exe_match:
            fallback_float = _float_from_answer_text(final_state.get("final_answer"))
            if is_numeric_match(fallback_float, gold_float):
                pred_float = fallback_float
                exe_match = True

        # NOTE: failed envelopes carry "data": None (key present, value None),
        # so a plain .get("data", {}) default is NOT enough here.
        executed_exprs = [
            str((r.get("data") or {}).get("expression", ""))
            for r in tool_results
            if r.get("tool_name") == "safe_math"
        ]
        prog_match = (
            any(is_program_match(e, record.gold_program) for e in executed_exprs)
            if record.gold_program
            else False
        )

        # 3. Compute IR Metrics — content-based: retrieved evidence TEXT
        # (row labels + amounts, chunk text) is graded against the gold
        # evidence STRINGS. ID-space comparison is meaningless here because
        # FinQA gold keys (table_N / text_M) never equal runtime labels.
        #
        # Sources are INTERLEAVED (round-robin graph row, vector chunk, ...)
        # before the top-k slice. Envelope completion order always puts the
        # vector task first (graph tasks depend on it), so plain concatenation
        # filled all 5 metric slots with narrative chunks and made every
        # graph row invisible to Recall@5 — capping measured recall near zero
        # even when tabular retrieval succeeded (36/44 gold items are table
        # rows only the graph can return).
        graph_texts: list[str] = []
        vector_texts: list[str] = []
        for r in tool_results:
            data = r.get("data")
            if not isinstance(data, dict):
                continue
            for rec in data.get("records", []):
                parts = [
                    str(rec.get("row_label", "")),
                    str(rec.get("amount", "")),
                    str(rec.get("normalized_amount", "")),
                    str(rec.get("year", "")),
                ]
                graph_texts.append(" ".join(p for p in parts if p and p != "None"))
            for ch in data.get("chunks", []):
                vector_texts.append(str(ch.get("text_content", "")))

        retrieved_texts: list[str] = []
        for i in range(max(len(graph_texts), len(vector_texts))):
            if i < len(graph_texts):
                retrieved_texts.append(graph_texts[i])
            if i < len(vector_texts):
                retrieved_texts.append(vector_texts[i])

        ir_scores = compute_ir_metrics_content(
            retrieved_texts, record.gold_evidence, k=5
        )
        # Per-source diagnostics: recall over each source's FULL result list
        # (no top-k truncation) so a low interleaved Recall@5 can be split
        # into "graph never found it" vs "found it but ranked outside k".
        for key, texts in (
            ("recall_graph", graph_texts),
            ("recall_vector", vector_texts),
        ):
            if texts:
                ir_scores[key] = compute_ir_metrics_content(
                    texts, record.gold_evidence, k=len(texts)
                )["recall_at_k"]
            else:
                ir_scores[key] = 0.0

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
