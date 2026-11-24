"""Bounded parsing and execution of FinQA arithmetic programs."""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
from hashlib import sha256

_ARITHMETIC = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
    "exp": operator.pow,
    "greater": operator.gt,
}
_TABLE = {"table_max", "table_min", "table_sum", "table_average"}
_OPERATIONS = set(_ARITHMETIC) | _TABLE
_COMMUTATIVE = {"add", "multiply"}
_CALL = re.compile(r"([a-z_]+)\s*\((.*)\)", re.I | re.S)
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?%?")
_REFERENCE = re.compile(r"#(\d+)")
_INFIX = {
    ast.Add: "add",
    ast.Sub: "subtract",
    ast.Mult: "multiply",
    ast.Div: "divide",
    ast.Pow: "exp",
}


@dataclass(frozen=True, slots=True)
class _Node:
    name: str
    args: tuple[_Node, ...] = ()
    literal: float | str | None = None


def _split(source: str, separators: str) -> list[str]:
    parts = []
    depth = 0
    start = 0
    quote = ""
    escaped = False
    for index, char in enumerate(source):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in {"'", '"'} and (index == 0 or source[index - 1] in "(, "):
            quote = char
        elif char == "(":
            depth += 1
            if depth > 24:
                raise ValueError("Program nesting exceeds the limit")
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced program")
        elif depth == 0 and char in separators:
            parts.append(source[start:index].strip())
            start = index + 1
    if depth or quote:
        raise ValueError("Unbalanced program")
    return [*parts, source[start:].strip()]


def _numeric_argument(value: str) -> float:
    source = value.strip().replace(",", "")
    if source.startswith("const_"):
        source = source.removeprefix("const_")
        if source == "m1":
            source = "-1"
    if not _NUMBER.fullmatch(source):
        raise ValueError("Unrecognized numeric argument")
    percentage = source.endswith("%")
    number = float(source.rstrip("%"))
    if not math.isfinite(number):
        raise ValueError("Nonfinite operand")
    return number / 100.0 if percentage else number


def _parse_operand(value: str, steps: list[_Node]) -> _Node:
    reference = _REFERENCE.fullmatch(value)
    if reference:
        index = int(reference.group(1))
        if index >= len(steps):
            raise ValueError("Program reference must point to an earlier step")
        return steps[index]
    if _CALL.fullmatch(value):
        return _parse_call(value, steps)
    return _Node("number", literal=_numeric_argument(value.lower()))


def _parse_call(source: str, steps: list[_Node]) -> _Node:
    match = _CALL.fullmatch(source)
    if not match or (name := match.group(1).lower()) not in _OPERATIONS:
        raise ValueError("Unrecognized operation")
    argument_source = match.group(2)
    if name in _TABLE:
        label, separator, unused = argument_source.rpartition(",")
        label = label.strip()
        unused = unused.strip()
        if not separator or not label or not unused:
            raise ValueError(
                "A table operation requires a row label and a second argument"
            )
        if label.startswith(("'", '"')):
            label = ast.literal_eval(label)
        if not isinstance(label, str):
            raise ValueError("The table row label must be text")
        return _Node(name, (_Node("table_auxiliary", literal=unused),), literal=label)
    args = _split(argument_source, ",")
    if len(args) != 2:
        raise ValueError("An arithmetic operation requires two operands")
    return _Node(name, tuple(_parse_operand(arg, steps) for arg in args))


def _parse_infix(node: ast.AST) -> _Node:
    if isinstance(node, ast.BinOp) and type(node.op) in _INFIX:
        return _Node(
            _INFIX[type(node.op)], (_parse_infix(node.left), _parse_infix(node.right))
        )
    if isinstance(node, ast.Name):
        return _Node("number", literal=_numeric_argument(node.id))
    value = ast.literal_eval(node)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("Expected a numeric literal")
    return _Node("number", literal=_numeric_argument(str(value)))


def _parse(source: str) -> list[_Node]:
    if not source or len(source) > 4096:
        raise ValueError("Missing or oversized program")
    parts = _split(source.strip(), ",;")
    if parts[-1] in {"EOF", ""}:
        parts.pop()
    if not parts or len(parts) > 64:
        raise ValueError("Invalid program length")
    if len(parts) == 1 and not _CALL.fullmatch(parts[0]):
        return [_parse_infix(ast.parse(parts[0], mode="eval").body)]
    steps: list[_Node] = []
    for part in parts:
        steps.append(_parse_call(part, steps))
    return steps


def canonicalize_program(source: str) -> str | None:
    """Resolve step references and compare the terminal expression structurally."""
    try:
        steps = _parse(source)
    except (ValueError, SyntaxError, TypeError, OverflowError, RecursionError):
        return None
    fingerprints: dict[int, str] = {}

    def fingerprint(node: _Node) -> str:
        key = id(node)
        if key not in fingerprints:
            children = [fingerprint(child) for child in node.args]
            if node.name in _COMMUTATIVE:
                children.sort()
            content = repr((node.name, node.literal, children)).encode()
            fingerprints[key] = sha256(content).hexdigest()
        return fingerprints[key]

    return fingerprint(steps[-1])


def _table_amount(raw: str) -> float:
    return _numeric_argument(raw.replace("$", "").split("(", 1)[0].strip())


def execute_program(source: str, table: list[list[str]]) -> float | str | None:
    """Run the terminal FinQA calculation and round as the official evaluator does."""
    rows = {row[0]: row[1:] for row in table if row}
    results: dict[int, float | str] = {}

    def execute(node: _Node) -> float | str:
        key = id(node)
        if key in results:
            return results[key]
        if node.name == "number":
            assert isinstance(node.literal, float)
            result: float | str = node.literal
        elif node.name in _TABLE:
            assert isinstance(node.literal, str)
            numbers = [_table_amount(raw) for raw in rows[node.literal]]
            if node.name == "table_min":
                result = min(numbers)
            elif node.name == "table_max":
                result = max(numbers)
            elif node.name == "table_average":
                result = sum(numbers) / len(numbers)
            else:
                result = sum(numbers)
        else:
            left, right = (execute(child) for child in node.args)
            if not isinstance(left, float) or not isinstance(right, float):
                raise ValueError("Arithmetic inputs must be numeric")
            computed = _ARITHMETIC[node.name](left, right)
            if isinstance(computed, bool):
                result = "yes" if computed else "no"
            elif isinstance(computed, float) and math.isfinite(computed):
                result = computed
            else:
                raise ValueError("Unexpected calculation output")
        if isinstance(result, float) and not math.isfinite(result):
            raise ValueError("Nonfinite result")
        results[key] = result
        return result

    try:
        steps = _parse(source)
        for step in steps:
            result = execute(step)
    except (
        ArithmeticError,
        ValueError,
        SyntaxError,
        KeyError,
        TypeError,
        RecursionError,
    ):
        return None
    return round(result, 5) if isinstance(result, float) else result
