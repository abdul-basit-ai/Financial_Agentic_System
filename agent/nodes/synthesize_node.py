"""Synthesis node validating evidence completeness and formatting citations.

Two synthesis paths share one evidence gate:
- LLM synthesizer (synthesizer_v2): analyst-prose answer over the verified
  evidence bundle, with a strict anti-hallucination post-check — every numeral
  in the LLM output must exist in the evidence (or the question). Any
  ungrounded figure rejects the LLM draft and falls back to the template.
- Deterministic template: used when no LLM is configured, the call fails, the
  output is ungrounded, or evidence is partial — evaluation and CI stay
  runnable offline.
"""

from __future__ import annotations

from typing import Any

from agent.llm import extract_numerals, invoke_llm, is_llm_enabled
from agent.nodes.relevance import question_terms, relevance_hits
from agent.prompts import load_prompt
from agent.state.schema import AgentStateV1

MAX_GRAPH_ITERATIONS = 3

_SYNTH_PROMPT_NAME = "synthesizer_v2"


def _to_float(token: str) -> float | None:
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def _strip_list_markers(text: str) -> str:
    """Removes bullet ordinals ("1.", "2)") so list formatting is not
    mistaken for ungrounded figures by the numeral check."""
    import re

    return re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)


def _is_grounded(answer: str, grounding_corpus: list[str]) -> bool:
    """True when every numeral in the answer exists in the corpus.

    The corpus includes the user question (quoted figures are user-provided,
    not hallucinations) plus every evidence string verbatim.
    """
    allowed: set[float] = set()
    for text in grounding_corpus:
        for token in extract_numerals(text):
            f = _to_float(token)
            if f is not None:
                allowed.add(f)

    for token in extract_numerals(_strip_list_markers(answer)):
        f = _to_float(token)
        if f is None:
            continue
        if f not in allowed:
            return False
    return True


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
            env
            for tid, env in state.sub_task_results.items()
            if isinstance(env, dict) and (tid, env.get("tool_name")) not in seen
        ]
        return state.tool_results + parallel_only

    all_results = _all_envelopes()
    math_results = [
        r for r in all_results if r.get("tool_name") == "safe_math" and r.get("success")
    ]
    graph_results = [
        r
        for r in all_results
        if r.get("tool_name") == "graph_retrieval" and r.get("success")
    ]
    vector_results = [
        r
        for r in all_results
        if r.get("tool_name") == "vector_retrieval" and r.get("success")
    ]

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

    # Relevance RANKING: evidence sharing signal with the question is surfaced
    # first, so citations and template answers draw from the matching rows.
    # Uses the shared relevance helpers (same term normalization as compute's
    # value picking: plural-folded, year-aware). Records/chunks with zero
    # overlap are DEMOTED, not discarded — exact-token filtering used to drop
    # correct evidence on vocabulary mismatches (revenue/revenues) and empty
    # envelopes flipped well-retrieved records into "insufficient evidence"
    # loops. Parroting unrelated rows is a precision concern; the grounding
    # gate and ranking handle it at far lower cost than recall loss.
    q_terms = question_terms(state.input)

    def _hits(text: str) -> int:
        return relevance_hits(text, q_terms)

    def _rank_graph_records(r: dict[str, Any]) -> dict[str, Any]:
        data = r.get("data") or {}
        records = data.get("records", [])
        if not q_terms:
            return r
        ranked = sorted(
            records,
            key=lambda rec: _hits(
                f"{rec.get('row_label', '')} {rec.get('column_header', '')}"
            ),
            reverse=True,
        )
        filtered = dict(r)
        filtered["data"] = {**data, "records": ranked}
        return filtered

    def _rank_vector_chunks(r: dict[str, Any]) -> dict[str, Any]:
        data = r.get("data") or {}
        chunks = data.get("chunks", [])
        if not q_terms:
            return r
        ranked = sorted(
            chunks,
            key=lambda ch: _hits(ch.get("text_content", "")),
            reverse=True,
        )
        filtered = dict(r)
        filtered["data"] = {**data, "chunks": ranked}
        return filtered

    graph_results = [_rank_graph_records(r) for r in graph_results]
    vector_results = [_rank_vector_chunks(r) for r in vector_results]

    has_sufficient_evidence = bool(math_results or graph_results or vector_results)
    failed_results = [r for r in state.tool_results if not r.get("success")]

    # Termination check: Sufficient data OR hit iteration limit
    if has_sufficient_evidence:
        grounding_corpus, evidence_block = _build_evidence_bundle(
            state, math_results, graph_results, vector_results
        )

        llm_answer, llm_log = _synthesize_with_llm(
            state, evidence_block, grounding_corpus
        )
        if llm_answer is not None:
            return {
                "final_answer": llm_answer,
                "is_terminal": True,
                "scratchpad": [
                    llm_log,
                    f"[Synthesizer {prompt_version} (hash:{prompt_hash})] "
                    f"LLM synthesis grounded on {len(grounding_corpus) - 1} evidence items.",
                ],
            }

        final_answer = _deterministic_answer(
            state, math_results, graph_results, vector_results
        )
        log_entry = (
            f"[Synthesizer {prompt_version} (hash:{prompt_hash}) rules] "
            f"Template synthesis over verified evidence (LLM path "
            f"{'unavailable' if not is_llm_enabled() else 'rejected/ungrounded'})."
        )
        if llm_log:
            log_entry = f"{llm_log}\n{log_entry}"
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


def _build_evidence_bundle(
    state: AgentStateV1,
    math_results: list[dict[str, Any]],
    graph_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
) -> tuple[list[str], str]:
    """Renders the verified evidence for the LLM prompt.

    Returns (grounding_corpus, formatted_block): the corpus is every string a
    synthesized numeral is allowed to match (question + evidence), the block
    is the prompt-ready text.
    """
    corpus: list[str] = [state.input]
    lines: list[str] = []

    if math_results:
        lines.append("Calculated Metrics (deterministic, already computed):")
        for m in math_results:
            d = m.get("data", {})
            entry = f"• Formula: {d.get('expression')} = {d.get('formatted')}"
            lines.append(entry)
            corpus.append(
                f"{d.get('expression')} {d.get('formatted')} {d.get('result')}"
            )

    if graph_results:
        lines.append("Verified Filing Data (table line items):")
        for g in graph_results:
            records = g.get("data", {}).get("records", [])
            # Relevance ranking upstream puts matching rows first; 10 slots
            # keep the matched row's sibling columns visible for per-ratio
            # questions ("payments volume" AND "transactions" live in one row).
            for rec in records[:10]:
                column = rec.get("column_header")
                column_part = f" [{column}]" if column else ""
                entry = (
                    f"• {rec.get('company')} ({rec.get('year') or 'FY'}): "
                    f"{rec.get('row_label')}{column_part} = {rec.get('amount')} "
                    f"(Normalized: {rec.get('normalized_amount')})"
                )
                lines.append(entry)
                corpus.append(
                    f"{rec.get('company')} {rec.get('year')} {rec.get('row_label')} "
                    f"{column or ''} {rec.get('amount')} {rec.get('normalized_amount')}"
                )

    if vector_results:
        lines.append("Disclosed Drivers & Context (narrative):")
        for v in vector_results:
            chunks = v.get("data", {}).get("chunks", [])
            for ch in chunks[:2]:
                entry = f"• [{ch.get('section')}] {ch.get('text_content')}"
                lines.append(entry)
                corpus.append(f"{ch.get('section')} {ch.get('text_content')}")

    return corpus, "\n".join(lines)


def _synthesize_with_llm(
    state: AgentStateV1,
    evidence_block: str,
    grounding_corpus: list[str],
) -> tuple[str | None, str]:
    """Attempts LLM synthesis. Returns (answer, log); answer None on any
    fallback condition (disabled, failure, empty, ungrounded numerals)."""
    if not is_llm_enabled():
        return None, ""

    system_prompt = load_prompt(_SYNTH_PROMPT_NAME)[0]
    user_prompt = (
        f"User question: {state.input}\n\n"
        f"VERIFIED EVIDENCE (the only allowed source of figures):\n"
        f"{evidence_block}\n\n"
        "Write the final answer. Bullet points only — no numbered lists, "
        "no markdown tables, no code fences."
    )

    result = invoke_llm(
        system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=900
    )
    if result is None:
        return None, ""

    log = (
        f"[Synthesizer llm:{result.model}] tokens {result.prompt_tokens}+"
        f"{result.completion_tokens}, est. cost ${result.cost_usd:.6f}."
    )

    answer = str(result.content or "").strip()
    if not answer:
        return None, f"{log} Draft empty; rejected."

    if not _is_grounded(answer, grounding_corpus):
        return (
            None,
            f"{log} Draft contained ungrounded numerals; rejected (no-hallucination policy).",
        )

    return answer, log


def _deterministic_answer(
    state: AgentStateV1,
    math_results: list[dict[str, Any]],
    graph_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
) -> str:
    """Template synthesis — the deterministic fallback renderer."""
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
            for rec in records[:10]:
                column = rec.get("column_header")
                column_part = f" [{column}]" if column else ""
                parts.append(
                    f"• {rec.get('company')} ({rec.get('year') or 'FY'}): "
                    f"{rec.get('row_label')}{column_part} = {rec.get('amount')} "
                    f"(Normalized: {rec.get('normalized_amount')})"
                )

    if vector_results:
        parts.append("\nDisclosed Drivers & Context:")
        for v in vector_results:
            chunks = v.get("data", {}).get("chunks", [])
            for ch in chunks[:2]:
                parts.append(f"• [{ch.get('section')}] {ch.get('text_content')}")

    return "\n".join(parts)


def _insufficient_evidence_answer(
    state: AgentStateV1, failed_results: list[dict[str, Any]]
) -> str:
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
            parts.append(
                f"• {r.get('tool_name')} (task {r.get('task_id')}): {r.get('error') or 'no matching data found'}"
            )
    parts.append(
        "\nSuggestion: verify the company identifier, fiscal years, or metric "
        "naming and try again."
    )
    return "\n".join(parts)
