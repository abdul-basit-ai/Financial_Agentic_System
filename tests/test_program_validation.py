import importlib

from agent.llm import LLMCallResult
from agent.nodes.program_validation import (
    execute_grounded_expression,
    program_validation_flags,
    requires_derived_program,
)
from agent.nodes.synthesize_node import _verify_program_candidates
from agent.state.schema import AgentStateV1


def test_derived_question_detection() -> None:
    assert requires_derived_program("what was the percentage change in revenue?")
    assert not requires_derived_program("what was total revenue in 2019?")


def test_program_flags_same_operand_ratio() -> None:
    flags = program_validation_flags(
        "what percentage of sales were international sales?",
        "divide(2455.0, 2455.0)",
    )
    assert "same_operand_reused" in flags
    assert "unit_ratio" in flags


def test_grounded_expression_rejects_invented_literal() -> None:
    assert (
        execute_grounded_expression("divide(20, 7)", "what is the ratio?", {20.0})
        is None
    )


def test_grounded_expression_executes_allowed_literals() -> None:
    result = execute_grounded_expression(
        "divide(subtract(120, 100), 100)", "what percentage change?", {120.0, 100.0}
    )
    assert result is not None
    assert result[0] == 0.2


def test_verifier_selects_grounded_candidate(monkeypatch) -> None:
    synthesize_module = importlib.import_module("agent.nodes.synthesize_node")
    response = LLMCallResult(
        content='{"selected_index": 1, "corrected_expression": null}',
        model="test-verifier",
        prompt_tokens=10,
        completion_tokens=5,
        cost_usd=0.01,
    )
    monkeypatch.setattr(synthesize_module, "invoke_llm", lambda **_: response)
    result = _verify_program_candidates(
        AgentStateV1(input="what ratio was reported?"),
        "Question: what ratio was reported?",
        ["divide(10, 10)", "divide(20, 10)"],
        {10.0, 20.0},
    )
    assert result is not None
    assert result[0] == 2.0
    assert result[1] == "divide(20, 10)"
