"""Synthesis node validating evidence completeness and formatting citations.

Three synthesis paths share one evidence gate:
- Table-conditioned program generation (program_gen_v1): when the anchored
  filing's FULL table was extracted and no explicit plan math ran, the LLM
  writes a FinQA-style program over table cells; the cells are resolved
  deterministically and executed via safe_math. This replaces value-fragment
  reassembly with the benchmark's native reasoning shape. The program runs
  against verified filing cells only; note it is computed at synthesis time,
  downstream of the risk gate's evidence review.
- LLM synthesizer (synthesizer_v2): analyst-prose answer over the verified
  evidence bundle, with a strict anti-hallucination post-check — every numeral
  in the LLM output must exist in the evidence (or the question). Any
  ungrounded figure rejects the LLM draft and falls back to the template.
- Deterministic template: used when no LLM is configured, the call fails, the
  output is ungrounded, or evidence is partial — evaluation and CI stay
  runnable offline.
"""

from __future__ import annotations

import re
from typing import Any

from agent.llm import extract_numerals, invoke_llm, is_llm_enabled
from agent.nodes.relevance import question_terms, relevance_hits
from agent.prompts import load_prompt
from agent.state.schema import AgentStateV1
from agent.tools.safe_math import SafeMathInput, safe_math_tool

MAX_GRAPH_ITERATIONS = 3

_SYNTH_PROMPT_NAME = "synthesizer_v2"
_PROGRAM_PROMPT_NAME = "program_gen_v1"

_CELL_REF_RE = re.compile(r"cell\(\s*[\"']([^\"']+)[\"']\s*,\s*[\"']([^\"']+)[\"']\s*\)")


def _norm_label(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _render_table_markdown(table_data: dict[str, Any]) -> str:
    """Renders the extracted table as markdown for the program generator."""
    rows = table_data.get("rows") or []
    headers: list[str] = list(table_data.get("headers") or [])
    for row in rows:
        for cell in row.get("cells") or []:
            header = str(cell.get("column_header") or "")
            if header and header not in headers:
                headers.append(header)
    if not rows:
        return "(empty table)"

    lines = ["| row label | " + " | ".join(headers) + " |"]
    lines.append("|" + "---|" * (len(headers) + 1))
    for row in rows:
        values: dict[str, str] = {}
        for cell in row.get("cells") or []:
            header = str(cell.get("column_header") or "")
            amount = cell.get("amount")
            values[header] = "" if amount is None else str(amount)
        lines.append(
            "| " + str(row.get("row_label") or "") + " | " + " | ".join(values.get(h, "") for h in headers) + " |"
        )
    return "\n".join(lines)


def _resolve_cell(table_data: dict[str, Any], row_ref: str, col_ref: str) -> float | None:
    """Resolves a cell(row, col) reference against the extracted table.

    Matching: exact label (case/whitespace-folded), then containment, then
    unique term-overlap best. Returns None when the reference is unresolvable
    or ambiguous — the caller fails the program rather than guessing.
    """
    rows = table_data.get("rows") or []
    row_norm = _norm_label(row_ref)
    col_norm = _norm_label(col_ref)

    target_rows = [r for r in rows if _norm_label(r.get("row_label")) == row_norm]
    if not target_rows:
        target_rows = [
            r
            for r in rows
            if row_norm and (row_norm in _norm_label(r.get("row_label")) or _norm_label(r.get("row_label")) in row_norm)
        ]
    if not target_rows:
        return None

    cells: list[dict[str, Any]] = []
    for r in target_rows:
        cells.extend(r.get("cells") or [])
    cells = [c for c in cells if c.get("amount") is not None]

    for c in cells:
        if _norm_label(c.get("column_header")) == col_norm:
            return float(c["amount"])
    for c in cells:
        header_norm = _norm_label(c.get("column_header"))
        if header_norm and (col_norm in header_norm or header_norm in col_norm):
            return float(c["amount"])

    c_terms = question_terms(f"{row_ref} {col_ref}")
    if not c_terms:
        return None
    best: dict[str, Any] | None = None
    best_score = 0
    ambiguous = False
    for c in cells:
        score = relevance_hits(
            f"{c.get('column_header', '')}", c_terms
        ) + relevance_hits(f"{target_rows[0].get('row_label', '')}", c_terms)
        if score > best_score:
            best, best_score, ambiguous = c, score, False
        elif score == best_score and score > 0:
            ambiguous = True
    if best is not None and not ambiguous:
        return float(best["amount"])
    return None


def _resolve_cells(program: str, table_data: dict[str, Any]) -> str:
    """Replaces every cell("row", "col") reference with its numeric value."""

    def _substitute(match: re.Match[str]) -> str:
        value = _resolve_cell(table_data, match.group(1), match.group(2))
        if value is None:
            raise ValueError(f"unresolvable cell reference: {match.group(0)}")
        return repr(value)

    return _CELL_REF_RE.sub(_substitute, program)


def _generate_and_execute_program(
    state: AgentStateV1, table_env: dict[str, Any]
) -> dict[str, Any] | None:
    """Generates a FinQA program over the full table and executes it.

    Self-consistency: samples the program PROGRAM_SELF_CONSISTENCY_SAMPLES
    times (default 5) at PROGRAM_SAMPLE_TEMPERATURE (default 0.7), executes
    every sample, and answers with the MAJORITY numeric result — operand-
    order and wrong-cell mistakes a single sample makes are outvoted (AAPL's
    percent change flipped sign in exactly one sample of five).

    Returns the terminal synthesis result, or None on any failure (no LLM,
    no sample parseable, unresolvable cells, math error) — callers fall back
    to the evidence-bundle synthesis path.
    """
    import os
    from collections import Counter

    prompt_text, prompt_version, prompt_hash = load_prompt(_PROGRAM_PROMPT_NAME)
    table_data = table_env.get("data") or {}
    table_md = _render_table_markdown(table_data)
    user_prompt = f"Question: {state.input}\n\nTable:\n{table_md}"

    try:
        k_samples = max(1, int(os.getenv("PROGRAM_SELF_CONSISTENCY_SAMPLES", "5")))
    except ValueError:
        k_samples = 5
    try:
        sample_temperature = float(os.getenv("PROGRAM_SAMPLE_TEMPERATURE", "0.7"))
    except ValueError:
        sample_temperature = 0.7

    from agent.nodes.plan_node import _extract_json

    def _one_sample(sample_idx: int) -> tuple[float, str] | None:
        """One LLM sample -> resolved program -> executed value, or None."""
        result = invoke_llm(
            system_prompt=prompt_text,
            user_prompt=user_prompt,
            max_tokens=600,
            temperature=sample_temperature if k_samples > 1 else None,
        )
        if result is None:
            return None
        parsed = _extract_json(result.content)
        program = parsed.get("program") if isinstance(parsed, dict) else None
        if not program or not isinstance(program, str):
            return None
        try:
            resolved = _resolve_cells(program, table_data)
        except ValueError:
            return None
        math_result = safe_math_tool(SafeMathInput(expression=resolved))
        if not math_result.success or math_result.data is None:
            return None
        return float(math_result.data.result), resolved

    if k_samples > 1:
        from concurrent.futures import ThreadPoolExecutor

        workers = min(k_samples, 4)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(_one_sample, range(k_samples)))
    else:
        outcomes = [_one_sample(0)]

    successful = [s for s in outcomes if s is not None]
    if not successful:
        print("[synthesize] program mode: no sample produced a computable program", flush=True)
        return None

    # Majority vote on values rounded to 6 decimals: identical programs
    # yield bit-identical floats, different-but-equivalent programs may
    # differ in the last ulp.
    votes = Counter(round(value, 6) for value, _ in successful)
    winning_key, winning_count = votes.most_common(1)[0]
    winner = next(s for s in successful if round(s[0], 6) == winning_key)
    answer_value, resolved = winner

    log = (
        f"[ProgramGen {prompt_version} (hash:{prompt_hash}) self-consistency "
        f"{winning_count}/{k_samples} samples agree; llm outcomes "
        f"{len(successful)}/{k_samples} computable.]"
    )
    print(f"[synthesize] {log} expr={resolved}", flush=True)

    final_text = (
        f"Computed from the filing table: {resolved} = {answer_value}\n"
        f"Answer: {answer_value}"
    )
    # safe_math-shaped envelope: feeds the evaluator's program-accuracy
    # metric (expression compared against the gold FinQA DSL) and keeps the
    # audit trail complete for a computation performed at synthesis time.
    program_envelope = {
        "task_id": "task_program",
        "tool_name": "safe_math",
        "success": True,
        "data": {
            "expression": resolved,
            "result": answer_value,
            "formatted": str(answer_value),
        },
        "error": None,
    }
    return {
        "final_answer": final_text,
        "is_terminal": True,
        "tool_results": [program_envelope],
        "scratchpad": [
            log,
            f"[ProgramGen] Executed '{resolved}' -> {answer_value} "
            f"({winning_count}/{k_samples} sample majority)",
        ],
    }



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
        if r.get("tool_name") == "table_extract":
            return bool(d.get("rows"))
        return False

    math_results = [r for r in math_results if _has_payload(r)]
    graph_results = [r for r in graph_results if _has_payload(r)]
    vector_results = [r for r in vector_results if _has_payload(r)]
    table_envs = [
        r
        for r in all_results
        if r.get("tool_name") == "table_extract"
        and r.get("success")
        and _has_payload(r)
    ]

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
        # TABLE-CONDITIONED PROGRAM MODE: full table extracted — generate a
        # FinQA program over the table and execute it. This takes priority
        # over planner-scheduled math: the program sees every row/column of
        # the filing, while compute's fragment picking resolves operands
        # heuristically (the JPM CET1 ratio computed 0.039 from wrong rows
        # that way). Falls through to the evidence-bundle synthesis on any
        # program failure; deterministic mode (no LLM) keeps the old paths.
        if table_envs and is_llm_enabled():
            program_outcome = _generate_and_execute_program(state, table_envs[0])
            if program_outcome is not None:
                return program_outcome

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

    # Final answer line — a single parseable number, mirroring the LLM
    # synthesizer's "Answer:" contract so the evaluator extracts the value
    # from the marker instead of scanning the whole dump (citation figures
    # and years otherwise masquerade as the prediction). Only emitted for a
    # REAL computed result: without math, no single row is "the" answer and
    # promoting the top row's amount fabricates a figure.
    if math_results:
        last = math_results[-1].get("data", {})
        answer_value = last.get("result")
        if answer_value is not None:
            parts.append(f"Answer: {answer_value}")

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
    parts.append("Answer: none")
    return "\n".join(parts)
