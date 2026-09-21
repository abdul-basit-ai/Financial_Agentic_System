"""Deterministic arithmetic node resolving parameters from parallel sub-task results."""

from __future__ import annotations

import re
from typing import Any

from agent.nodes.relevance import best_matching_record, question_terms, relevance_hits
from agent.state.schema import AgentStateV1
from agent.tools.safe_math import SafeMathInput, safe_math_tool


def _record_amount(record: dict[str, Any]) -> float | None:
    amount = record.get("amount")
    if amount is None:
        amount = record.get("normalized_amount")
    try:
        return float(amount) if amount is not None else None
    except (TypeError, ValueError):
        return None


def _pick_numeric_value(
    records: list[dict[str, Any]],
    q_terms: set[str],
    task_year: int | None = None,
    metric_hint: str | None = None,
) -> float | None:
    """Extracts the amount the dispatching task actually asked for.

    Scoping order (each stage keeps its narrowing only when non-empty):
    1. Task fiscal year — matched on Value.year or a year-as-row-label
       (FinQA maturity schedules put years in row labels).
    2. Task METRIC hint — picks the COLUMN (row label + column header
       overlap with the hint's terms). Without this, a multi-operand plan
       ("payment volume" and "transactions") resolves BOTH task_N.amount
       references to the same top-relevance record: divide(x, x) = 1.0.
    3. Question terms — picks the ROW (the company/line the question is
       about among same-column values).
    4. Question-term best match / first record (previous behavior).
    """
    if not records:
        return None

    candidates = records

    if task_year is not None:
        year_str = str(task_year)
        matched = [
            rec
            for rec in candidates
            if rec.get("year") == task_year
            or str(rec.get("row_label", "")).strip() == year_str
        ]
        if matched:
            candidates = matched

    if metric_hint:
        m_terms = question_terms(metric_hint)
        if m_terms:
            hits = [
                relevance_hits(
                    f"{rec.get('row_label', '')} {rec.get('column_header', '')}", m_terms
                )
                for rec in candidates
            ]
            top = max(hits)
            if top > 0:
                candidates = [rec for rec, h in zip(candidates, hits) if h == top]

    row_hits = [relevance_hits(str(rec.get("row_label", "")), q_terms) for rec in candidates]
    top_row = max(row_hits)
    if top_row > 0:
        candidates = [rec for rec, h in zip(candidates, row_hits) if h == top_row]

    if len(candidates) == 1:
        return _record_amount(candidates[0])

    best = best_matching_record(candidates, q_terms)
    if best is None:
        return None
    return _record_amount(best)


def _resolve_dynamic_expression(
    expression: str,
    tool_results: list[dict[str, Any]],
    sub_task_results: dict[str, Any] | None = None,
    q_terms: set[str] | None = None,
    task_years: dict[str, int | None] | None = None,
    task_metrics: dict[str, str | None] | None = None,
) -> str:
    """Substitutes dynamic task references (e.g., task_1.amount) with concrete values."""
    resolved = expression
    pattern = re.compile(r"\b(task_\d+)\.amount\b")
    matches = pattern.findall(expression)
    q_terms = q_terms or set()
    task_years = task_years or {}
    task_metrics = task_metrics or {}

    # Build lookup map: task_id -> numeric amount
    lookup: dict[str, float] = {}

    def _value_from(data: Any, task_year: int | None, metric_hint: str | None) -> float | None:
        if not isinstance(data, dict):
            return None
        records = data.get("records", [])
        if records:
            value = _pick_numeric_value(records, q_terms, task_year, metric_hint)
            if value is not None:
                return value
        result = data.get("result")
        try:
            return float(result) if result is not None else None
        except (TypeError, ValueError):
            return None

    # Check parallel sub_task_results first
    if sub_task_results:
        for t_id, envelope in sub_task_results.items():
            value = (
                _value_from(envelope.get("data"), task_years.get(t_id), task_metrics.get(t_id))
                if isinstance(envelope, dict)
                else None
            )
            if value is not None:
                lookup[t_id] = value

    # Supplement with sequential tool_results
    for res in tool_results:
        res_id: Any = res.get("task_id")
        if not res_id or res_id in lookup:
            continue
        value = _value_from(
            res.get("data"), task_years.get(str(res_id)), task_metrics.get(str(res_id))
        )
        if value is not None:
            lookup[res_id] = value

    for m in matches:
        if m in lookup:
            resolved = resolved.replace(f"{m}.amount", str(lookup[m]))

    return resolved


def compute_node(state: AgentStateV1) -> dict[str, Any]:
    """Executes safe arithmetic calculations using AST evaluator."""
    remaining_calls: list[dict[str, Any]] = []
    executed_results: list[dict[str, Any]] = []
    sub_task_updates: dict[str, Any] = {}
    scratchpad_logs: list[str] = []

    # Map each retrieval task to the fiscal year AND metric hint it was
    # dispatched for, so math references pick the value tied to THAT task's
    # year and COLUMN instead of an arbitrary record from the result set.
    task_years: dict[str, int | None] = {}
    task_metrics: dict[str, str | None] = {}
    for call in state.tool_calls:
        if call.get("target_tool") in {"graph_retrieval", "vector_retrieval"}:
            payload = call.get("payload", {})
            year = payload.get("year")
            metric = payload.get("metric_name")
            task_years[str(call.get("task_id"))] = (
                int(year) if year is not None else None
            )
            task_metrics[str(call.get("task_id"))] = str(metric) if metric else None

    for call in state.tool_calls:
        if call.get("target_tool") == "safe_math" and call.get("status") == "PENDING":
            raw_expr = call.get("payload", {}).get("expression", "")
            concrete_expr = _resolve_dynamic_expression(
                raw_expr,
                state.tool_results,
                state.sub_task_results,
                q_terms=question_terms(state.input),
                task_years=task_years,
                task_metrics=task_metrics,
            )

            # If dynamic references remain unresolved, the dependency tasks
            # produced no usable values (e.g. retrieval found nothing). Fail
            # THIS task cleanly instead of calling safe_math and crashing with
            # a confusing ASTSecurityError on 'task_2.amount'.
            unresolved = re.findall(r"\btask_\d+\.amount\b", concrete_expr)
            if unresolved:
                error_msg = (
                    f"Cannot evaluate expression: no numeric data for "
                    f"{', '.join(sorted(set(unresolved)))} (dependency retrieval "
                    f"returned no records)."
                )
                task_id = call.get("task_id", "task_math")
                executed_results.append(
                    {
                        "task_id": task_id,
                        "tool_name": "safe_math",
                        "success": False,
                        "data": None,
                        "error": error_msg,
                    }
                )
                scratchpad_logs.append(
                    f"[Safe Math] Skipped '{raw_expr}' -> {error_msg}"
                )
                continue

            result = safe_math_tool(SafeMathInput(expression=concrete_expr))
            task_id = call.get("task_id", "task_math")

            envelope = {
                "task_id": task_id,
                "tool_name": "safe_math",
                "success": result.success,
                "data": result.data.model_dump() if result.data else None,
                "error": result.error,
            }
            executed_results.append(envelope)
            sub_task_updates[task_id] = envelope

            val = result.data.formatted if result.data else "ERROR"
            scratchpad_logs.append(
                f"[Safe Math] Evaluated '{concrete_expr}' -> Result: {val}"
            )
        else:
            remaining_calls.append(dict(call))

    return {
        "tool_calls": remaining_calls,
        "tool_results": executed_results,
        "sub_task_results": sub_task_updates,
        "scratchpad": scratchpad_logs,
    }
