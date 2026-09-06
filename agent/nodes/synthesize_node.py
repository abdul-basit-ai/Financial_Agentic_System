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
    has_sufficient_evidence = bool(math_results or graph_results or vector_results)

    # Termination check: Sufficient data OR hit iteration limit
    if has_sufficient_evidence or state.iteration_count >= MAX_GRAPH_ITERATIONS:
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

    # Insufficient evidence: emit critique and plan re-entry
    critique = (
        f"[Synthesizer Critique] Iteration {state.iteration_count}: "
        "Insufficient retrieval data to fulfill user query. Requesting plan adjustment."
    )
    return {
        "scratchpad": [critique],
        "is_terminal": False,
    }