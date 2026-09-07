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
    """Calculates multi-hop information retrieval ranking metrics."""
    gold_set = {g.strip().lower() for g in gold_items if g.strip()}
    if not gold_set:
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