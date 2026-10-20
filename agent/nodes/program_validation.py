"""Deterministic safeguards for generated financial arithmetic programs."""

from __future__ import annotations

import re

from agent.tools.safe_math import SafeMathInput, safe_math_tool

_NUMBER_RE = re.compile(r"(?<![A-Za-z_])-?\d+(?:\.\d+)?")
_FUNCTION_RE = re.compile(r"\b(add|subtract|multiply|divide)\s*\(")


def requires_derived_program(question: str) -> bool:
    """Whether a question normally needs more than a direct cell lookup."""
    text = question.casefold()
    return any(
        marker in text
        for marker in (
            "average",
            "average change",
            "per ",
            "ratio",
            "portion",
            "percentage",
            "percent",
            "change",
            "increase",
            "decrease",
            "decline",
            "growth",
            "how much",
            "market capitalization",
        )
    )


def program_validation_flags(question: str, expression: str) -> list[str]:
    """Returns suspicious-pattern flags without rejecting valid programs alone."""
    flags: list[str] = []
    numbers = _NUMBER_RE.findall(expression)
    lowered = question.casefold()
    if len(numbers) >= 2 and len(set(numbers)) == 1:
        if any(
            word in lowered for word in ("ratio", "percentage", "portion", "change")
        ):
            flags.append("same_operand_reused")
    if "subtract(" in expression and len(numbers) >= 2 and numbers[0] == numbers[1]:
        flags.append("zero_subtraction")
    if "divide(" in expression and len(numbers) >= 2 and numbers[0] == numbers[1]:
        flags.append("unit_ratio")
    if (
        any(word in lowered for word in ("percentage", "percent", "ratio", "portion"))
        and "divide(" not in expression
    ):
        flags.append("missing_division")
    if (
        any(word in lowered for word in ("average", "avg"))
        and "divide(" not in expression
    ):
        flags.append("missing_average_division")
    if (
        any(word in lowered for word in ("increase", "growth", "change"))
        and "subtract(" not in expression
    ):
        flags.append("missing_difference")
    return flags


def execute_grounded_expression(
    expression: str, question: str, allowed_values: set[float]
) -> tuple[float, list[str]] | None:
    """Executes only safe arithmetic whose literals occur in evidence/question."""
    flags = program_validation_flags(question, expression)
    for token in _NUMBER_RE.findall(expression):
        value = float(token)
        # Standard financial constants are allowed for percentage conversions.
        if value not in allowed_values and value not in {1.0, 2.0, 3.0, 100.0, 1000.0}:
            return None
    result = safe_math_tool(SafeMathInput(expression=expression))
    if not result.success or result.data is None:
        return None
    return float(result.data.result), flags
