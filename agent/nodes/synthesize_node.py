"""Synthesis node validating evidence completeness and formatting citations."""

from __future__ import annotations

from typing import Any

from agent.prompts import load_prompt
from agent.state.schema import AgentStateV1

MAX_GRAPH_ITERATIONS = 3


def synthesize_node(state: AgentStateV1) -> dict[str, Any]:
    """Audits tool results and synthesizes grounded financial response."""
    _, prompt_version, prompt_hash = load_prompt("synthesizer_v1")

    # Check for completed tool evidence from BOTH sequential (tool_results)
    # and parallel (sub_task_results) execution paths.
    def _all_envelopes() -> list[dict[str, Any]]:
        seen = {
            (r.get("task_id"), r.get("tool_name"))
            for r in state.tool_results
            if r.get("task_id")
        }
        parallel_only = [
            env for tid, env in state.sub_task_results.items()
            if (tid, env.get("tool_name")) not in seen and isinstance(env, dict)
        ]
        return state.tool_results + parallel_only

    all_results = _all_envelopes()
    math_results = [r for r in all_results if r.get("tool_name") == "safe_math" and r.get("success")]
    graph_results = [r for r in all_results if r.get("tool_name") == "graph_retrieval" and r.get("success")]
    vector_results = [r for r in all_results if r.get("tool_name") == "vector_retrieval" and r.get("success")]

    # Verification: Do we have sufficient data to answer?
    # A tool can exit cleanly (success=True) yet return zero records — that is
    # NOT evidence. Only non-empty payloads count.
    def _has_payload(r: dict[str, Any]) -> bool:
        d = r.get("data") or {}
        if r.get("tool_name") == "safe_math":
            return d.get("result") is not None
        if r.get("tool_name") == "graph_retrieval":
            return bool(d.get("records"))
        if r.get("tool_name") == "vector_retrieval":
            return bool(d.get("chunks"))
        return False

    math_results = [r for r in math_results if _has_payload(r)]
    graph_results = [r for r in graph_results if _has_payload(r)]
    vector_results = [r for r in vector_results if _has_payload(r)]

    # Relevance filter: retrieved evidence must share signal with the question.
    # Without this, unrelated rows (e.g. leftover test fixtures) get parroted
    # as "verified" citations for a question they don't answer.
    def _question_terms(text: str) -> set[str]:
        stop = {
            "what", "was", "the", "in", "of", "for", "and", "a", "an", "to",
            "is", "were", "on", "by", "with", "from", "at", "did", "how",
            "much", "many", "that", "this", "company", "fiscal", "year",
        }
        return {
            w for w in str(text).lower().replace("%", " ").replace(",", " ").split()
            if len(w) > 2 and w not in stop and not w.isdigit()
        }

    q_terms = _question_terms(state.input)

    def _relevance_hits(evidence_text: str) -> int:
        ev_terms = _question_terms(evidence_text)
        return len(q_terms & ev_terms)

    # Graph records: keep only rows whose label overlaps the question
    # (e.g. question mentions "goodwill" -> keep "goodwill" rows), unless the
    # question has no usable terms at all.
    def _filter_graph_records(r: dict[str, Any]) -> dict[str, Any]:
        data = r.get("data") or {}
        records = data.get("records", [])
        if not q_terms:
            return r
        relevant = [
            rec for rec in records
            if _relevance_hits(f"{rec.get('row_label', '')} {rec.get('company', '')}") >= 1
        ]
        filtered = dict(r)
        filtered["data"] = {**data, "records": relevant}
        return filtered

    # Vector chunks: similarity threshold already applied at retrieval; here
    # require at least one question-term overlap in the chunk text.
    def _filter_vector_chunks(r: dict[str, Any]) -> dict[str, Any]:
        data = r.get("data") or {}
        chunks = data.get("chunks", [])
        if not q_terms:
            return r
        relevant = [
            ch for ch in chunks
            if _relevance_hits(ch.get("text_content", "")) >= 1
        ]
        filtered = dict(r)
        filtered["data"] = {**data, "chunks": relevant}
        return filtered

    graph_results = [_filter_graph_records(r) for r in graph_results]
    graph_results = [r for r in graph_results if r["data"].get("records")]
    vector_results = [_filter_vector_chunks(r) for r in vector_results]
    vector_results = [r for r in vector_results if r["data"].get("chunks")]

    has_sufficient_evidence = bool(math_results or graph_results or vector_results)
    failed_results = [r for r in state.tool_results if not r.get("success")]

    # Termination check: Sufficient data OR hit iteration limit
    if has_sufficient_evidence:
        # Build synthesis response
        parts = [f"Financial Analysis for: {state.input}\n"]

        if math_results:
            parts.append("Calculated Metrics:")
            for m in math_results:
                d = m.get("data", {})
                parts.append(f"• Formula: {d.get('expression')} = {d.get('formatted')}")

        if graph_results:
            parts.append("\nVerified Filing Data:")
            for g in graph_results:
                records = g.get("data", {}).get("records", [])
                for rec in records[:5]:
                    parts.append(
                        f"• {rec.get('company')} ({rec.get('year') or 'FY'}): "
                        f"{rec.get('row_label')} = {rec.get('amount')} (Normalized: {rec.get('normalized_amount')})"
                    )

        if vector_results:
            parts.append("\nDisclosed Drivers & Context:")
            for v in vector_results:
                chunks = v.get("data", {}).get("chunks", [])
                for ch in chunks[:2]:
                    parts.append(f"• [{ch.get('section')}] {ch.get('text_content')}")

        final_answer = "\n".join(parts)
        log_entry = (
            f"[Synthesizer {prompt_version} (hash:{prompt_hash})] "
            f"Successfully synthesized response with {len(state.tool_results)} verified evidence items."
        )

        return {
            "final_answer": final_answer,
            "is_terminal": True,
            "scratchpad": [log_entry],
        }

    # Hit iteration limit with no evidence: terminate honestly, don't loop
    if state.iteration_count >= MAX_GRAPH_ITERATIONS:
        final_answer = _insufficient_evidence_answer(state, failed_results)
        log_entry = (
            f"[Synthesizer {prompt_version} (hash:{prompt_hash})] "
            f"Terminated at iteration limit with insufficient evidence "
            f"({len(failed_results)} failed steps)."
        )
        return {
            "final_answer": final_answer,
            "is_terminal": True,
            "scratchpad": [
                log_entry,
                f"[Synthesizer] No viable evidence after iteration {state.iteration_count}; answer marked incomplete.",
            ],
        }

    # Insufficient evidence: emit critique and plan re-entry
    critique = (
        f"[Synthesizer Critique] Iteration {state.iteration_count}: "
        "Insufficient retrieval data to fulfill user query. Requesting plan adjustment."
    )
    return {
        "scratchpad": [critique],
        "is_terminal": False,
    }


def _insufficient_evidence_answer(state: AgentStateV1, failed_results: list[dict[str, Any]]) -> str:
    """Builds an explicit 'cannot answer' response per the synthesizer prompt's
    Completeness Verification principle — never fabricate figures."""
    parts = [f"Financial Analysis for: {state.input}\n"]
    parts.append(
        "Insufficient evidence: I could not retrieve enough verified data to "
        "answer this question. Per the no-hallucination policy, no figures are provided."
    )
    if failed_results:
        parts.append("\nWhat failed:")
        for r in failed_results[:5]:
            parts.append(f"• {r.get('tool_name')} (task {r.get('task_id')}): {r.get('error') or 'no matching data found'}")
    parts.append(
        "\nSuggestion: verify the company identifier, fiscal years, or metric "
        "naming and try again."
    )
    return "\n".join(parts)