"""Shared question-evidence term overlap helpers.

Used by the synthesizer (evidence relevance filtering) and the compute node
(picking which retrieved record feeds a dynamic task_N.amount reference), so
both nodes apply the same notion of "signal overlap" and stay consistent.
"""

from __future__ import annotations

_QUESTION_STOPWORDS = frozenset({
    "what", "was", "the", "in", "of", "for", "and", "a", "an", "to",
    "is", "were", "on", "by", "with", "from", "at", "did", "how",
    "much", "many", "that", "this", "company", "fiscal", "year",
})


def question_terms(text: str) -> set[str]:
    """Content terms of a question/evidence string for overlap scoring.

    Drops stopwords, punctuation-separated tokens shorter than 3 chars, and
    pure digits (amounts match via the numeric machinery elsewhere).
    """
    return {
        w for w in str(text).lower().replace("%", " ").replace(",", " ").split()
        if len(w) > 2 and w not in _QUESTION_STOPWORDS and not w.isdigit()
    }


def relevance_hits(evidence_text: str, q_terms: set[str]) -> int:
    """Number of question terms present in the evidence text."""
    return len(q_terms & question_terms(evidence_text))


def best_matching_record(
    records: list[dict],
    q_terms: set[str],
) -> dict | None:
    """Picks the record whose row_label best overlaps the question terms.

    Falls back to the first record when no label overlaps (or the question
    has no usable terms) — the previous always-first behavior.
    """
    if not records:
        return None
    if not q_terms:
        return records[0]

    best: dict | None = None
    best_score = 0
    for rec in records:
        score = relevance_hits(str(rec.get("row_label", "")), q_terms)
        if score > best_score:
            best, best_score = rec, score
    return best if best is not None else records[0]
