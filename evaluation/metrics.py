"""FinQA execution values, program structure, and evidence ranking metrics."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any

from ingestion.unit_normalizer import UNIT_MULTIPLIERS, parse_amount

_NUMBER = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
_SCALE = r"(?:\s+(?:thousands?|millions?|billions?|trillions?))?"
_FINANCIAL_TOKEN = re.compile(
    rf"(?<!\w)(?:[$€£¥]\s*)?(?:\(\s*{_NUMBER}\s*%?{_SCALE}\s*\)\s*%?|{_NUMBER}\s*%?{_SCALE})(?!\w)",
    re.I,
)
_CITATION = re.compile(r"\[[^\[\]\r\n]+\]")
_ANSWER_MARKER = re.compile(r"\banswer\s*:", re.I)
_ABSTENTION = re.compile(
    r"\b(?:cannot|can.t|unable|insufficient|unknown|abstain)\b", re.I
)
_STOP_WORDS = frozenset(
    "the a an of is was were in for and to at on by with as it its s".split()
)


def parse_float_safe(value: Any) -> float | None:
    """Parse a single literal amount without accepting surrounding prose."""
    amount, _, _ = parse_amount(value)
    return amount


def _answer_literal(value: Any) -> tuple[float | str | None, bool, str | None]:
    if value is None or isinstance(value, bool):
        return None, False, None
    if isinstance(value, int | float):
        return parse_amount(value)

    text = _CITATION.sub("", str(value)).strip()
    markers = list(_ANSWER_MARKER.finditer(text))
    if markers:
        text = text[markers[-1].end() :].strip().split("\n", 1)[0]
    text = text.strip().strip("*`").rstrip(".").strip()
    text = re.sub(r"\s+\(percent\)$", "%", text, flags=re.I)
    if _ABSTENTION.search(text):
        return None, False, None
    if text.lower() in {"yes", "no"}:
        return text.lower(), False, None
    parsed = parse_amount(text)
    if parsed[0] is not None:
        return parsed

    tokens = list(_FINANCIAL_TOKEN.finditer(text))
    if len(tokens) == 1 and tokens[0].end() == len(text):
        return parse_amount(tokens[0].group())
    return None, False, None


def parse_answer_value(value: Any) -> float | str | None:
    """Read the displayed final answer, converting an explicit percentage to a fraction."""
    amount, percentage, scale = _answer_literal(value)
    if isinstance(amount, float):
        if percentage:
            return amount / 100.0
        if scale is not None:
            return amount * UNIT_MULTIPLIERS[scale]
    return amount


def parse_answer_float(value: Any) -> float | None:
    pred = parse_answer_value(value)
    return pred if isinstance(pred, float) else None


def execution_value(value: Any) -> float | str | None:
    if isinstance(value, str) and value.strip().lower() in {"yes", "no"}:
        return value.strip().lower()
    amount, percentage, scale = parse_amount(value)
    if amount is None:
        return None
    if percentage:
        return amount / 100.0
    return amount * UNIT_MULTIPLIERS.get(scale or "", 1.0)


def is_numeric_match(pred: Any, gold: Any) -> bool:
    """Match the FinQA evaluator's five-decimal execution result against its gold value."""
    predicted = execution_value(pred)
    expected = execution_value(gold)
    if isinstance(predicted, float) and isinstance(expected, float):
        return (
            math.isfinite(predicted)
            and math.isfinite(expected)
            and round(predicted, 5) == expected
        )
    return isinstance(predicted, str) and predicted == expected


def is_display_match(
    pred: Any, gold: Any, *, abs_tol: float = 0.0, rel_tol: float = 0.0
) -> bool:
    """Optional display score against the dataset's presentation answer."""
    predicted, pct_pred, scale_pred = _answer_literal(pred)
    expected, pct_gold, scale_gold = _answer_literal(gold)
    if pct_pred != pct_gold:
        return False
    if isinstance(predicted, float) and isinstance(expected, float):
        p = predicted * UNIT_MULTIPLIERS.get(scale_pred or "", 1.0)
        g = expected * UNIT_MULTIPLIERS.get(scale_gold or "", 1.0)
        return math.isclose(p, g, abs_tol=abs_tol, rel_tol=rel_tol)
    return isinstance(predicted, str) and predicted == expected


def evidence_tokens(
    text: str,
) -> tuple[list[tuple[float, bool, str | None]], list[str]]:
    normalized = str(text).lower().replace("\u2212", "-")
    numbers = []
    for match in _FINANCIAL_TOKEN.finditer(normalized):
        value, pct, scale = parse_amount(match.group().strip())
        if value is not None:
            numbers.append((value, pct, scale))
    words = [
        token
        for token in re.findall(r"[a-z]+", _FINANCIAL_TOKEN.sub(" ", normalized))
        if token not in _STOP_WORDS
    ]
    return numbers, words


def evidence_hit(gold_text: str, retrieved_text: str) -> bool:
    """Require the full gold numeric sequence and its significant words."""
    gold_numbers, gold_words = evidence_tokens(gold_text)
    retrieved_numbers, retrieved_words = evidence_tokens(retrieved_text)
    remaining = iter(retrieved_numbers)
    return (
        bool(gold_numbers or gold_words)
        and all(number in remaining for number in gold_numbers)
        and set(gold_words).issubset(retrieved_words)
    )


def ranked_metrics(
    hits_by_rank: list[set[int]], gold_count: int, k: int
) -> dict[str, float]:
    if not gold_count:
        return {"recall_at_k": 1.0, "precision_at_k": 1.0, "mrr": 1.0, "ndcg_at_k": 1.0}

    covered: set[int] = set()
    reciprocal_rank = 0.0
    dcg = 0.0
    relevant = 0
    top_k = hits_by_rank[:k]
    for rank, hits in enumerate(top_k, start=1):
        if hits - covered:
            covered.update(hits)
            relevant += 1
            reciprocal_rank = reciprocal_rank or (1.0 / rank)
            dcg += 1.0 / math.log2(rank + 1)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, gold_count) + 1))
    return {
        "recall_at_k": round(len(covered) / gold_count, 4),
        "precision_at_k": round(relevant / max(1, len(top_k)), 4),
        "mrr": round(reciprocal_rank, 4),
        "ndcg_at_k": round(dcg / idcg, 4) if idcg else 0.0,
    }


def _content_metrics(
    retrieved: list[str],
    gold: list[str] | set[str],
    k: int,
    match: Callable[[str, str], bool],
) -> dict[str, float]:
    gold_items = list(
        dict.fromkeys(item.strip().lower() for item in gold if item.strip())
    )
    hits_by_rank = [
        {
            index
            for index, gold_item in enumerate(gold_items)
            if match(gold_item, retrieved_item)
        }
        for retrieved_item in retrieved[:k]
    ]
    return ranked_metrics(hits_by_rank, len(gold_items), k)


def compute_ir_metrics(
    retrieved_items: list[str], gold_items: list[str] | set[str], k: int = 5
) -> dict[str, float]:
    return _content_metrics(
        retrieved_items, gold_items, k, lambda gold, item: gold == item.strip().lower()
    )


def compute_ir_metrics_content(
    retrieved_texts: list[str], gold_texts: list[str] | set[str], k: int = 5
) -> dict[str, float]:
    return _content_metrics(retrieved_texts, gold_texts, k, evidence_hit)


def is_program_match(pred_prog: str, gold_prog: str) -> bool:
    from evaluation.finqa_program import canonicalize_program

    predicted = canonicalize_program(pred_prog)
    return predicted is not None and predicted == canonicalize_program(gold_prog)


def canonicalize_program(prog_str: str) -> Any:
    from evaluation.finqa_program import canonicalize_program as canonicalize

    return canonicalize(prog_str)
