import json

import pytest

from agent.documents import reasoning
from agent.documents.models import Evidence
from agent.llm import LLMCallResult


def cell(key, row="Revenue", period="2023", amount=10.0, **overrides):
    values = dict(
        evidence_id=key,
        document_id="r",
        filename="ABC Annual Report.pdf",
        document_hash="digest",
        page=4,
        text=f"{row} | {period} | {amount} USD million",
        table_id="p4:t0",
        row_index=1,
        column_index=2,
        row_label=row,
        column_label=period,
        raw_value=str(amount),
        numeric_value=amount,
        normalized_amount=amount * 1e6,
        unit="million",
        scale=1e6,
        currency="USD",
        quantity_kind="currency",
        period=period,
    )
    return Evidence.model_validate({**values, **overrides})


def plan(op, *ids):
    return {"operation": op, "operand_ids": list(ids)}


def completion(data):
    return LLMCallResult(json.dumps(data), "fake", 10, 5, None)


def test_lookup_exact_quantity_currency_and_requested_scale(monkeypatch):
    monkeypatch.setattr(reasoning, "is_llm_enabled", lambda: False)
    income = cell("income", row="Net income", amount=100.0)
    revenue = cell("rev")
    answer = reasoning.answer_question(
        "What was revenue in USD millions in 2023?", [income, revenue]
    )
    assert answer["answer_status"] == "grounded"
    assert answer["calculation"]["result"] == 10.0
    assert answer["calculation"]["unit"] == "USD millions"
    assert [i["evidence_id"] for i in answer["citations"]] == ["rev"]
    assert "p. 4" in answer["final_answer"]


@pytest.mark.parametrize(
    "bad",
    [
        plan("lookup", "income"),
        plan("multiply", "rev", "income"),
        plan("lookup", "other-year"),
        plan("lookup", "nonexistent"),
        {"operation": ["lookup"]},
        plan("lookup", "missing-period"),
        plan("lookup", "rev", "rev"),
        {"operation": True},
    ],
)
def test_provider_cannot_select_an_unrelated_cell_or_operation(bad):
    rows = [
        cell("rev"),
        cell("income", row="Operating income"),
        cell("other-year", period="2022"),
        cell("missing-period", period=None),
    ]
    output = reasoning.execute_plan("What was revenue in 2023?", bad, rows)
    assert output["answer_status"] == "insufficient"
    assert not output["citations"] and output["calculation"] is None


def test_ambiguous_matches_beyond_retrieval_window_are_rejected(monkeypatch):
    monkeypatch.setattr(reasoning, "is_llm_enabled", lambda: True)
    monkeypatch.setattr(
        reasoning, "invoke_llm", lambda *a, **kw: completion(plan("lookup", "000"))
    )
    rows = [cell(f"{n:03d}", page=n + 1) for n in range(25)]
    out = reasoning.answer_question("What was revenue in 2023?", rows)
    assert out["answer_status"] == "insufficient"
    assert len(out["evidence"]) == 24
    assert out["llm_usage"][0]["model"] == "fake"


def test_duplicate_ids_conflicting_units_zero_divisor_and_normalization():
    a, b = cell("a", period="2022", amount=0.0), cell("b", period="2023")
    q = "What was the percentage change in revenue from 2022 to 2023?"
    p = plan("percent_change", "a", "b")
    assert reasoning.execute_plan(q, p, [a, b])["answer_status"] == "insufficient"
    a = cell("a", period="2022", amount=5.0)
    for replacement in [
        b.model_copy(update={"currency": "EUR"}),
        b.model_copy(update={"units_ambiguous": True}),
        b.model_copy(update={"numeric_value": float("inf")}),
        b.model_copy(update={"normalized_amount": 77.0}),
        b.model_copy(update={"scale": 0.0}),
        b.model_copy(update={"numeric_value": None}),
    ]:
        assert (
            reasoning.execute_plan(q, p, [a, replacement])["answer_status"]
            == "insufficient"
        )
    assert reasoning.execute_plan(q, p, [a, b, a])["answer_status"] == "insufficient"
    assert (
        reasoning.execute_plan(q, plan("percent_change", "a", "a"), [a, b])[
            "answer_status"
        ]
        == "insufficient"
    )


def test_time_direction_and_distinct_denominator_for_change():
    old, new = cell("old", period="2022", amount=8.0), cell("new", period="2023")
    q = "What was the percentage change in revenue from 2022 to 2023?"
    valid = reasoning.execute_plan(q, plan("percent_change", "old", "new"), [new, old])
    assert valid["calculation"]["result"] == 25.0
    assert "**25%**" in valid["final_answer"]
    assert (
        reasoning.execute_plan(q, plan("percent_change", "new", "old"), [new, old])[
            "answer_status"
        ]
        == "insufficient"
    )
    earlier = "What was the percentage change in revenue from 2023 to 2022?"
    reverse = reasoning.execute_plan(
        earlier, plan("percent_change", "new", "old"), [old, new]
    )
    assert reverse["calculation"]["result"] == -20.0


def test_absolute_change_addition_and_explicit_ratio_dimensions():
    old, new = cell("old", period="2022", amount=8.0), cell("new", period="2023")
    assert (
        reasoning.execute_plan(
            "What was the change in revenue from 2022 to 2023?",
            plan("subtract", "new", "old"),
            [old, new],
        )["calculation"]["result"]
        == 2e6
    )
    cost = cell("cost", row="Expenses", amount=2.0)
    p = plan("add", "cost", "new")
    assert (
        reasoning.execute_plan(
            "What was the sum of revenue and expenses in 2023?", p, [new, cost]
        )["calculation"]["result"]
        == 12e6
    )
    ratio = reasoning.execute_plan(
        "What was revenue divided by expenses in 2023?",
        plan("divide", "new", "cost"),
        [new, cost],
    )
    assert (
        ratio["calculation"]["unit"] == "ratio"
        and ratio["calculation"]["result"] == 5.0
    )
    rev_percent = reasoning.execute_plan(
        "What percentage of revenue was expenses in 2023?",
        plan("percent_of", "cost", "new"),
        [new, cost],
    )
    assert rev_percent["calculation"]["result"] == 20.0


def test_table_lookup_percent_is_rendered_with_percent_units():
    margin = cell(
        "m",
        row="Operating margin",
        amount=12.5,
        scale=1.0,
        unit="percent",
        quantity_kind="percent",
        normalized_amount=12.5,
        currency=None,
    )
    answer = reasoning.execute_plan(
        "What was operating margin in 2023?", plan("lookup", "m"), [margin]
    )
    assert answer["calculation"]["result"] == 12.5
    assert "**12.5%**" in answer["final_answer"]


def test_unicode_and_reported_rounding_preserve_financial_digits():
    large = cell("a", amount=123.456789, scale=1e6, normalized_amount=123456789.0)
    answer = reasoning.execute_plan(
        "What was revenue in 2023?", plan("lookup", "a"), [large]
    )
    assert "123,456,789 USD" in answer["final_answer"]


@pytest.mark.parametrize("content", ["not JSON", "{}"])
def test_malformed_plan_costs_and_safe_provider_fallback(monkeypatch, content):
    monkeypatch.setattr(reasoning, "is_llm_enabled", lambda: True)
    monkeypatch.setattr(
        reasoning,
        "invoke_llm",
        lambda *a, **kw: LLMCallResult(content, "fake", 10, 5, None),
    )
    out = reasoning.answer_question("What was revenue in 2023?", [cell("a")])
    assert len(out["llm_usage"]) == 1
    assert out["llm_usage"][0]["cost_usd"] is None
    if content == "not JSON":
        assert out["calculation"]["result"] == 1e7
    else:
        assert out["answer_status"] == "insufficient"


def test_verbatim_quote_must_preserve_negation_and_receive_verification(monkeypatch):
    text = "The company did not declare bankruptcy."
    source = Evidence(
        evidence_id="b",
        document_id="r",
        filename="ABC.pdf",
        document_hash="hash",
        page=2,
        text=text,
    )
    question = "Did the company declare bankruptcy?"
    invalid = {
        "operation": "quotes",
        "quotes": [{"evidence_id": "b", "quote": "declare bankruptcy."}],
    }
    assert (
        reasoning.execute_plan(question, invalid, [source])["answer_status"]
        == "insufficient"
    )
    valid = {"operation": "quotes", "quotes": [{"evidence_id": "b", "quote": text}]}
    monkeypatch.setattr(reasoning, "is_llm_enabled", lambda: True)
    for supported in (None, False, True):
        monkeypatch.setattr(
            reasoning,
            "invoke_llm",
            lambda *a, task, **kw: (
                completion(valid)
                if task == "document_reasoning"
                else None if supported is None else completion({"supported": supported})
            ),
        )
        out = reasoning.answer_question(question, [source])
        assert (out["answer_status"] == "grounded") is bool(supported)
        assert [entry["stage"] for entry in out["llm_usage"]] == [
            "document_reasoning",
            "verifier",
        ]
        if supported:
            assert out["citations"][0]["quote"] == text


def test_annual_lookup_cannot_use_a_quarterly_cell():
    qtr = cell("q", period="Q1 2023")
    p = plan("lookup", "q")
    assert (
        reasoning.execute_plan("What was revenue in 2023?", p, [qtr])["answer_status"]
        == "insufficient"
    )


def test_unknown_monetary_unit_cannot_be_reported_as_a_count():
    r = cell(
        "r",
        quantity_kind="number",
        currency=None,
        unit="base",
        scale=1.0,
        normalized_amount=10.0,
    )
    assert (
        reasoning.execute_plan("What was revenue in 2023?", plan("lookup", "r"), [r])[
            "answer_status"
        ]
        == "insufficient"
    )


@pytest.mark.parametrize(
    "method,expected", [("ocr", "insufficient"), ("ocr_verified_table", "grounded")]
)
def test_ocr_arithmetic_requires_verified_table_geometry(method, expected):
    row = cell("r", extraction_method=method)
    assert (
        reasoning.execute_plan("What was revenue in 2023?", plan("lookup", "r"), [row])[
            "answer_status"
        ]
        == expected
    )


def test_explicit_division_of_periods_uses_requested_order():
    old, new = cell("old", period="2022", amount=8.0), cell("new", period="2023")
    result = reasoning.execute_plan(
        "What was the ratio of revenue in 2023 to 2022?",
        plan("divide", "new", "old"),
        [old, new],
    )
    assert result["calculation"]["result"] == 1.25


@pytest.mark.parametrize(
    "question",
    ["What was revenue per share in 2023?", "What was the revenue rate in 2023?"],
)
def test_currency_line_cannot_supply_a_rate_or_per_share_lookup(question):
    assert (
        reasoning.execute_plan(question, plan("lookup", "r"), [cell("r")])[
            "answer_status"
        ]
        == "insufficient"
    )
