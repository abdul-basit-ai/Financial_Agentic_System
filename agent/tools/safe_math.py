"""Safe mathematical and financial calculator using Python AST."""

from __future__ import annotations

import ast
import operator
import re
from typing import Any
from pydantic import BaseModel, Field

from agent.tools.base import ToolResult


class SafeMathInput(BaseModel):
    expression: str = Field(
        ...,
        description="Mathematical expression or FinQA DSL program, e.g. '(150.5 + 49.5) / 2' or 'divide(100, 20)'",
    )


class SafeMathOutput(BaseModel):
    expression: str = Field(..., description="Original input expression")
    result: float = Field(..., description="Computed numerical result")
    formatted: str = Field(..., description="Formatted string representation of result")


# Whitelisted binary and unary operators
SAFE_OPERATORS: dict[type[ast.operator | ast.unaryop], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Whitelisted FinQA DSL functions & table aggregations
SAFE_FUNCTIONS: dict[str, Any] = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
    "exp": operator.pow,
    "greater": lambda a, b: 1.0 if a > b else 0.0,
    "table_sum": lambda *args: float(sum(args)),
    "table_max": lambda *args: float(max(args)),
    "table_min": lambda *args: float(min(args)),
    "table_average": lambda *args: float(sum(args) / len(args)) if args else 0.0,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}

MAX_AST_DEPTH = 50
CURRENCY_STRIP_RE = re.compile(r"[\$€£¥]")
# Thousands separators inside numeric literals only: "1,200" / "1,200,000" -> digits joined.
# Requires digit-comma-digit grouping so argument commas in DSL calls are untouched.
THOUSANDS_SEP_RE = re.compile(r"(?<=\d),(?=\d\d\d(?!\d))")
PERCENT_SUB_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


class ASTSecurityError(ValueError):
    """Raised when an expression contains unapproved AST nodes."""


class SafeMathEvaluator:
    """Recursively evaluates whitelisted AST nodes with depth limits."""

    def __init__(self, max_depth: int = MAX_AST_DEPTH) -> None:
        self.max_depth = max_depth

    def _eval_node(self, node: ast.AST, current_depth: int = 0) -> float:
        if current_depth > self.max_depth:
            raise ASTSecurityError(f"Expression exceeds maximum recursion depth of {self.max_depth}")

        if isinstance(node, ast.Expression):
            return self._eval_node(node.body, current_depth + 1)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ASTSecurityError(f"Unsupported constant type: {type(node.value)}")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in SAFE_OPERATORS:
                raise ASTSecurityError(f"Unsupported binary operator: {op_type.__name__}")
            left = self._eval_node(node.left, current_depth + 1)
            right = self._eval_node(node.right, current_depth + 1)
            if op_type is ast.Div and right == 0.0:
                raise ZeroDivisionError("Division by zero in arithmetic expression")
            return float(SAFE_OPERATORS[op_type](left, right))

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in SAFE_OPERATORS:
                raise ASTSecurityError(f"Unsupported unary operator: {op_type.__name__}")
            operand = self._eval_node(node.operand, current_depth + 1)
            return float(SAFE_OPERATORS[op_type](operand))

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ASTSecurityError("Dynamic function dispatch is forbidden")
            func_name = node.func.id.lower()
            if func_name not in SAFE_FUNCTIONS:
                raise ASTSecurityError(f"Function '{func_name}' is not permitted")

            args = [self._eval_node(arg, current_depth + 1) for arg in node.args]
            if func_name == "divide" and len(args) >= 2 and args[1] == 0.0:
                raise ZeroDivisionError("Division by zero in FinQA divide() call")

            res = SAFE_FUNCTIONS[func_name](*args)
            return float(res)

        raise ASTSecurityError(f"Forbidden AST node detected: {type(node).__name__}")

    def evaluate(self, expr_str: str) -> float:
        clean_expr = expr_str.strip()
        if not clean_expr:
            raise ValueError("Empty mathematical expression")

        # Strip currency characters: $100 -> 100
        clean_expr = CURRENCY_STRIP_RE.sub("", clean_expr)

        # Strip thousands separators ONLY inside numeric literals ("1,200" -> "1200").
        # Removing commas globally would break FinQA DSL calls like divide(100, 4).
        clean_expr = THOUSANDS_SEP_RE.sub("", clean_expr)

        # Convert percentage notation: 15% -> (15 / 100)
        clean_expr = PERCENT_SUB_RE.sub(r"(\1 / 100)", clean_expr)

        try:
            tree = ast.parse(clean_expr, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Syntax error in expression: {exc}") from exc

        return self._eval_node(tree)


def evaluate_math(expression: str) -> SafeMathOutput:
    """Pure calculation function without instrumentation."""
    evaluator = SafeMathEvaluator()
    result = evaluator.evaluate(expression)
    formatted = f"{result:,.4f}".rstrip("0").rstrip(".") if "." in f"{result:,.4f}" else f"{result:,.0f}"
    return SafeMathOutput(expression=expression, result=result, formatted=formatted)


def safe_math_tool(payload: SafeMathInput) -> ToolResult[SafeMathOutput]:
    """Instrumented tool entrypoint for safe arithmetic."""
    return ToolResult.execute_instrumented(
        tool_name="safe_math",
        fn=evaluate_math,
        expression=payload.expression,
    )