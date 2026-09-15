"""Quantitative evaluation metrics engine for financial reasoning.

Implements:
1. Dynamic hybrid tolerance execution accuracy (Acc_exe) with scale equivalence.
2. Commutative AST isomorphism canonicalization for FinQA programs (Acc_prog).
3. Multi-hop IR metrics: Recall@K, Precision@K, MRR, and NDCG@K against gold_inds.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any

FLOAT_REGEX = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+")
COMMUTATIVE_OPS: set[str] = {"add", "multiply", "table_sum", "table_max", "table_min"}


def parse_float_safe(val: Any) -> float | None:
    """Safely extracts a floating-point number from numbers, strings, or currency text."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace("$", "").replace("€", "").replace("£", "").replace(",", "")
    if s.endswith("%"):
        s = s[:-1].strip()
    match = FLOAT_REGEX.search(s)
    if match:
        try:
            return float(match.group(0).replace(",", ""))
        except ValueError:
            return None
    return None


def is_numeric_match(
    pred: Any,
    gold: Any,
    abs_tol: float = 1e-3,
    rel_tol: float = 1e-2,
) -> bool:
    """Evaluates numerical execution accuracy with scale invariance (e.g. 0.15 vs 15%)."""
    p = parse_float_safe(pred)
    g = parse_float_safe(gold)

    if p is None or g is None:
        return False

    # Check 1: Direct comparison with dynamic tolerance: max(abs_tol, rel_tol * |gold|)
    tol = max(abs_tol, rel_tol * abs(g))
    if abs(p - g) <= tol:
        return True

    # Check 2: Scale mismatch (percentage vs ratio: 0.15 vs 15.0)
    if abs((p * 100.0) - g) <= max(abs_tol, rel_tol * abs(g)):
        return True
    if abs((p / 100.0) - g) <= max(abs_tol, rel_tol * abs(g)):
        return True

    return False


def _canonicalize_ast_node(node: ast.AST) -> Any:
    """Recursively converts AST nodes into canonical tuples, sorting commutative children."""
    if isinstance(node, ast.Expression):
        return _canonicalize_ast_node(node.body)

    if isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, (int, float)):
            return ("const", round(float(val), 6))
        return ("const", str(val))

    if isinstance(node, ast.Name):
        return ("name", node.id.lower())

    if isinstance(node, ast.Call):
        func_name = node.func.id.lower() if isinstance(node.func, ast.Name) else "unknown"
        child_canonical = [_canonicalize_ast_node(arg) for arg in node.args]

        if func_name in COMMUTATIVE_OPS:
            # Sort canonicalized children to achieve permutation invariance: add(a, b) == add(b, a)
            child_canonical = sorted(child_canonical, key=lambda x: str(x))

        return ("call", func_name, tuple(child_canonical))

    if isinstance(node, ast.BinOp):
        op_map = {
            ast.Add: "add",
            ast.Mult: "multiply",
            ast.Sub: "subtract",
            ast.Div: "divide",
            ast.Pow: "exp",
        }
        op_type = type(node.op)
        func_name = op_map.get(op_type, op_type.__name__.lower())
        left = _canonicalize_ast_node(node.left)
        right = _canonicalize_ast_node(node.right)
        children = [left, right]
        if func_name in COMMUTATIVE_OPS:
            children = sorted(children, key=lambda x: str(x))
        return ("call", func_name, tuple(children))

    if isinstance(node, ast.UnaryOp):
        op_name = "neg" if isinstance(node.op, ast.USub) else "pos"
        return ("unary", op_name, _canonicalize_ast_node(node.operand))

    return ("raw", ast.dump(node))


def canonicalize_program(prog_str: str) -> Any:
    """Parses and canonicalizes a FinQA DSL program string into an isomorphism tuple."""
    clean_prog = prog_str.strip().replace(",", ", ").replace(";", "")
    clean_prog = re.sub(r"\s+", " ", clean_prog)
    try:
        tree = ast.parse(clean_prog, mode="eval")
        return _canonicalize_ast_node(tree)
    except SyntaxError:
        # Fallback string normalization if not valid Python syntax
        return ("fallback", clean_prog.lower())


def is_program_match(pred_prog: str, gold_prog: str) -> bool:
    """Checks if predicted program is isomorphic to gold program under commutativity."""
    if not pred_prog or not gold_prog:
        return False
    canon_pred = canonicalize_program(pred_prog)
    canon_gold = canonicalize_program(gold_prog)
    return canon_pred == canon_gold


def compute_ir_metrics(
    retrieved_items: list[str],
    gold_items: list[str] | set[str],
    k: int = 5,
) -> dict[str, float]:
    """Calculates multi-hop information retrieval ranking metrics.

    ID-space variant: compares retrieved identifier strings against gold
    identifier strings for exact membership. Only meaningful when both sides
    share one ID vocabulary; FinQA gold_inds keys (table_N / text_M) do NOT
    match runtime row labels or chunk keys, so prefer compute_ir_metrics_content
    for benchmark runs over real data.
    """
    gold_set = {g.strip().lower() for g in gold_items if g.strip()}
    if not gold_set:
        # Records without gold evidence cannot be graded; returning the
        # neutral 1.0 convention keeps them from dragging means down, but
        # analysis should exclude them explicitly.
        return {"recall_at_k": 1.0, "precision_at_k": 1.0, "mrr": 1.0, "ndcg_at_k": 1.0}

    k_retrieved = [r.strip().lower() for r in retrieved_items[:k]]

    hits = sum(1 for item in k_retrieved if item in gold_set)
    recall = hits / len(gold_set)
    precision = hits / max(1, len(k_retrieved))

    # Mean Reciprocal Rank
    mrr = 0.0
    for idx, item in enumerate(k_retrieved, start=1):
        if item in gold_set:
            mrr = 1.0 / idx
            break

    # Normalized Discounted Cumulative Gain (NDCG)
    dcg = 0.0
    for idx, item in enumerate(k_retrieved, start=1):
        if item in gold_set:
            dcg += 1.0 / math.log2(idx + 1)

    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(gold_set)) + 1))
    ndcg = (dcg / idcg) if idcg > 0.0 else 0.0

    return {
        "recall_at_k": round(recall, 4),
        "precision_at_k": round(precision, 4),
        "mrr": round(mrr, 4),
        "ndcg_at_k": round(ndcg, 4),
    }


# =====================================================================
# Content-based IR metrics (real FinQA gold_inds format)
# =====================================================================
# FinQA gold_indices maps IDs like "table_3" / "text_34" to the HUMAN-READABLE
# evidence string they reference ("... payments volume ( billions ) is 637 ...").
# Runtime retrieval produces row labels + amounts and raw chunk text — a
# different ID vocabulary entirely, so exact key comparison is always zero.
# The content variant grades each gold evidence string by token overlap with
# retrieved text instead.

_EVIDENCE_STOPWORDS = frozenset({
    "the", "a", "an", "of", "is", "was", "were", "in", "for", "and", "to",
    "at", "on", "by", "with", "as", "it", "its", "s",
})


def evidence_tokens(text: str) -> tuple[set[str], set[str]]:
    """Splits text into (numeric_tokens, content_word_tokens) for overlap scoring.

    Numbers are significant in financial evidence (an amount match is strong
    signal), so they are tracked separately from words.
    """
    raw = re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)?", str(text).lower())
    numbers: set[str] = set()
    words: set[str] = set()
    for tok in raw:
        if any(ch.isdigit() for ch in tok):
            # Strip trailing ".0" float noise so 637.0 matches gold 637
            numbers.add(tok.rstrip("0").rstrip(".") if "." in tok else tok)
        elif len(tok) > 2 and tok not in _EVIDENCE_STOPWORDS:
            words.add(tok)
    return numbers, words


def evidence_hit(gold_text: str, retrieved_text: str) -> bool:
    """Whether a retrieved text covers a gold evidence string.

    Rule: a hit requires sharing at least one significant token of each kind
    the gold string carries — numbers (if any) and content words. A gold
    string without numbers (pure narrative) needs one content-word overlap.
    """
    g_num, g_words = evidence_tokens(gold_text)
    r_num, r_words = evidence_tokens(retrieved_text)

    number_ok = (not g_num) or bool(g_num & r_num)
    word_ok = (not g_words) or bool(g_words & r_words)
    # Pure-number gold (e.g. a lone value) still needs the number itself.
    return number_ok and word_ok and bool(g_num or (g_words & r_words))


def compute_ir_metrics_content(
    retrieved_texts: list[str],
    gold_texts: list[str] | set[str],
    k: int = 5,
) -> dict[str, float]:
    """IR metrics graded by content overlap instead of identifier equality.

    A retrieved item is *relevant* if it covers at least one gold evidence
    string (evidence_hit). Recall is over gold items covered by the top-k;
    precision over top-k items that are relevant; MRR/NDCG use the first/
    graded relevant positions.
    """
    gold_list = [g for g in gold_texts if str(g).strip()]
    if not gold_list:
        # Same neutral convention as compute_ir_metrics for missing gold.
        return {"recall_at_k": 1.0, "precision_at_k": 1.0, "mrr": 1.0, "ndcg_at_k": 1.0}

    top_k = [str(t) for t in retrieved_texts[:k] if str(t).strip()]

    # Binary relevance of each retrieved slot: covers any gold evidence?
    relevance: list[bool] = [
        any(evidence_hit(g, t) for g in gold_list) for t in top_k
    ]

    # Recall: fraction of gold items covered by ANY top-k retrieval
    covered = 0
    for g in gold_list:
        if any(evidence_hit(g, t) for t in top_k):
            covered += 1
    recall = covered / len(gold_list)

    precision = sum(relevance) / max(1, len(top_k))

    mrr = 0.0
    for idx, rel in enumerate(relevance, start=1):
        if rel:
            mrr = 1.0 / idx
            break

    # NDCG with per-gold credit: once a gold item is covered by an earlier
    # (higher-ranked) slot, later slots covering the SAME gold item add no
    # further gain — without this, duplicates inflate DCG above IDCG (>1.0).
    dcg = 0.0
    credited: set[int] = set()
    for idx, text in enumerate(top_k, start=1):
        for g_idx, g in enumerate(gold_list):
            if g_idx in credited:
                continue
            if evidence_hit(g, text):
                dcg += 1.0 / math.log2(idx + 1)
                credited.add(g_idx)
                break

    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(gold_list)) + 1))
    ndcg = (dcg / idcg) if idcg > 0.0 else 0.0

    return {
        "recall_at_k": round(recall, 4),
        "precision_at_k": round(precision, 4),
        "mrr": round(mrr, 4),
        "ndcg_at_k": round(ndcg, 4),
    }
