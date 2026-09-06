"""Synthesis node validating evidence completeness and formatting citations."""

from __future__ import annotations

from typing import Any

from agent.prompts import load_prompt
from agent.state.schema import AgentStateV1

MAX_GRAPH_ITERATIONS = 3


def synthesize_node(state: AgentStateV1) -> dict[str, Any]:
    """Audits tool results and synthesizes grounded financial response."""
    _, prompt_version, prompt_hash = load_prompt("synthesizer_v1")

    # Check for completed tool evidence
    math_results = [r for r in state.tool_results if r.get("tool_name") == "safe_math" and r.get("success")]
    graph_results = [r for r in state.tool_results if r.get("tool_name") == "graph_retrieval" and r.get("success")]
    vector_results = [r for r in state.tool_results if r.get("tool_name") == "vector_retrieval" and r.get("success")]

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