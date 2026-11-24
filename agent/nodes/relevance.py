"""Shared question-evidence term overlap helpers.

Used by the synthesizer (evidence relevance filtering) and the compute node
(picking which retrieved record feeds a dynamic task_N.amount reference), so
both nodes apply the same notion of "signal overlap" and stay consistent.
"""

from __future__ import annotations

_QUESTION_STOPWORDS = frozenset(
    {
        "what",
        "was",
        "the",
        "in",
        "of",
        "for",
        "and",
        "a",
        "an",
        "to",
        "is",
        "were",
        "on",
        "by",
        "with",
        "from",
        "at",
        "did",
        "how",
        "much",
        "many",
        "that",
        "this",
        "company",
        "fiscal",
        "year",
    }
)


def normalize_term(word: str) -> str:
    """Crude plural/possessive folding so 'payments' matches 'payment'.

    Exact morphology is unnecessary at this scale — evidence and question
    overwhelmingly differ by a trailing 's' (revenue/revenues,
    transaction/transactions) or an apostrophe (stockholders' / stockholders).
    """
    w = word.strip().lower().strip("!?.,;:()'\"")
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w


def question_terms(text: str) -> set[str]:
    """Content terms of a question/evidence string for overlap scoring.

    Drops stopwords and punctuation noise; KEEPS 4-digit years (a question
    asking about 2007 should match evidence tagged 2007). Terms are
    plural-folded via normalize_term on both sides of every comparison.
    """
    terms = set()
    for raw in str(text).lower().replace("%", " ").replace(",", " ").split():
        if not raw or raw in _QUESTION_STOPWORDS:
            continue
        norm = normalize_term(raw)
        if not norm:
            continue
        if norm.isdigit():

            if len(norm) == 4 and norm.startswith(("19", "20")):
                terms.add(norm)
            continue
        if len(norm) > 2:
            terms.add(norm)
    return terms


def relevance_hits(evidence_text: str, q_terms: set[str]) -> int:
    """Number of question terms present in the evidence text."""
    return len(q_terms & question_terms(evidence_text))


def best_matching_record(
    records: list[dict],
    q_terms: set[str],
) -> dict | None:
    """Return a unique positive match across row and column labels."""
    if not records or not q_terms:
        return None
    scores = [
        relevance_hits(
            f"{r.get('row_label', '')} {r.get('column_header', '')}", q_terms
        )
        for r in records
    ]
    top = max(scores)
    if not top or scores.count(top) != 1:
        return None
    return records[scores.index(top)]
