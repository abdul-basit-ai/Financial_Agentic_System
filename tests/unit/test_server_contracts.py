"""Boundary validation tests for the public HTTP schemas."""

import pytest
from pydantic import ValidationError

from agent.server.schemas import HITLApprovalRequest, QueryRequest


def test_query_request_trims_input_and_accepts_safe_identifiers() -> None:
    request = QueryRequest(
        query="  What was revenue?  ",
        company_identifier="  AAPL  ",
        thread_id="thread:demo-1",
        tenant_id="tenant_1",
    )

    assert request.query == "What was revenue?"
    assert request.company_identifier == "AAPL"


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "   "},
        {"query": "Revenue?", "thread_id": "../../unsafe"},
        {"query": "Revenue?", "unexpected": True},
    ],
)
def test_query_request_rejects_invalid_boundary_input(payload: dict) -> None:
    with pytest.raises(ValidationError):
        QueryRequest(**payload)


def test_hitl_request_rejects_blank_analyst_id() -> None:
    with pytest.raises(ValidationError):
        HITLApprovalRequest(action="APPROVE", analyst_id="   ")
