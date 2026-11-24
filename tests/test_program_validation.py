from agent.nodes.program_validation import (
    execute_grounded_expression,
    program_validation_flags,
    requires_derived_program,
)


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


def test_constants_are_not_evidence() -> None:
    assert (
        execute_grounded_expression("multiply(1000, 10)", "what is revenue?", {10})
        is None
    )
    assert (
        execute_grounded_expression("add(0xFF, 10)", "what is revenue?", {10}) is None
    )
