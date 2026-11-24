"""Resolve arithmetic operands to unique filing cells before evaluation."""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent.tools.safe_math import evaluate_math


def source_page(record: dict[str, Any]) -> int:
    page = record.get("page")
    if isinstance(page, int) and not isinstance(page, bool) and page > 0:
        return page
    for key in ("source_file", "filename", "record_id", "report_id"):
        match = re.search(
            r"(?:^|[/\\])page[_-](\d+)\b", str(record.get(key) or ""), re.I
        )
        if match and int(match[1]) > 0:
            return int(match[1])
    raise ValueError("No source page for this filing evidence")


def citation(record: dict[str, Any]) -> dict[str, Any]:
    document = record.get("record_id") or record.get("report_id")
    row = record.get("row_label")
    column = record.get("column_header") or record.get("year")
    if not document or not row or column is None:
        raise ValueError("A filing, row and column are required for numeric evidence")
    page = source_page(record)
    row_index = record.get("row_index")
    column_index = record.get("column_index")
    evidence_id = record.get("evidence_id") or (
        f"{document}::{record.get('table_id') or 'table'}::"
        f"{row_index if row_index is not None else row}::{column_index if column_index is not None else column}"
    )
    return {
        key: value
        for key, value in {
            **record,
            "document_id": document,
            "evidence_id": evidence_id,
            "page": page,
            "row_label": row,
            "column_header": str(column),
        }.items()
        if value is not None
    }


@dataclass
class Operand:
    expression: str
    citations: list[dict[str, Any]]
    unit: str
    scale: float

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Operand:
        if record.get("units_ambiguous"):
            raise ValueError("Ambiguous units")
        amount = record.get("amount")
        if (
            not isinstance(amount, (float, int))
            or isinstance(amount, bool)
            or not math.isfinite(amount)
        ):
            raise ValueError("Missing or nonfinite cell")
        scale = record.get("scale_multiplier")
        if (
            not isinstance(scale, (int, float))
            or not math.isfinite(scale)
            or scale <= 0
        ):
            raise ValueError("Unknown unit scale")
        unit = str(record.get("quantity_kind") or record.get("unit_label") or "")
        if not unit:
            raise ValueError("Unknown unit")
        if unit in {"currency", "per_share"}:
            unit = f"{unit} {record.get('currency') or record.get('unit_label') or 'unspecified'}"
        return cls(repr(float(amount)), [citation(record)], unit, float(scale))


_BINARY = {ast.Add: "add", ast.Sub: "subtract", ast.Mult: "multiply", ast.Div: "divide"}


def resolve_arithmetic(
    expression: str, resolve_reference: Callable[[ast.expr], Operand | None]
) -> Operand:
    if not expression or len(expression) > 4096:
        raise ValueError("Missing or oversized program")
    tree = ast.parse(expression.strip(), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError("Program exceeds operation limit")

    def visit(node: ast.expr) -> Operand:
        reference = resolve_reference(node)
        if reference is not None:
            return reference
        if isinstance(node, ast.BinOp):
            operator = _BINARY.get(type(node.op))
            args = [node.left, node.right]
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and not node.keywords
        ):
            operator = node.func.id
            args = node.args
        else:
            raise ValueError("An operand must reference evidence")
        if operator not in {
            "add",
            "subtract",
            "multiply",
            "divide",
            "table_average",
            "table_sum",
        }:
            raise ValueError("Unsupported arithmetic operation")
        if not args or (
            operator not in {"table_average", "table_sum"} and len(args) != 2
        ):
            raise ValueError("Invalid operand count")
        values = [visit(arg) for arg in args]
        first, last = values[0], values[-1]
        if operator in {"add", "subtract", "table_sum", "table_average"}:
            if any((v.unit, v.scale) != (first.unit, first.scale) for v in values):
                raise ValueError("Incompatible units or scales")
            unit, scale = first.unit, first.scale
        elif operator == "divide":
            if first.scale != last.scale:
                raise ValueError("Ratio requires compatible reporting scales")
            unit = "ratio" if first.unit == last.unit else f"{first.unit}/{last.unit}"
            scale = 1.0
        else:
            unit, scale = f"{first.unit} * {last.unit}", first.scale * last.scale
        citations = {
            str(c["evidence_id"]): c for value in values for c in value.citations
        }
        if len({c["document_id"] for c in citations.values()}) > 1:
            raise ValueError("The calculation spans multiple filing records")
        return Operand(
            f"{operator}({', '.join(v.expression for v in values)})",
            list(citations.values()),
            unit,
            scale,
        )

    return visit(tree.body)


def calculation_data(operand: Operand, program: str) -> dict[str, Any]:
    result = evaluate_math(operand.expression)
    return {
        **result.model_dump(),
        "program": program,
        "citations": operand.citations,
        "unit_label": operand.unit,
        "scale_multiplier": operand.scale,
    }
