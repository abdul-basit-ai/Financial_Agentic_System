"""Tests for schema export and persistent metrics logging."""

from __future__ import annotations

import json
import os

import pytest

from agent.tools import safe_math_tool, SafeMathInput
from agent.tools.metrics_logger import log_tool_result, read_metrics, summarize_metrics
from agent.tools.schema_export import export_schemas, load_tool_schema, SCHEMA_DIR

EXPECTED_TOOLS = {
    "safe_math",
    "graph_retrieval",
    "vector_retrieval",
    "context_fusion",
    "query_decomposition",
}


def test_export_schemas_writes_all_tools(tmp_path) -> None:
    out_dir = tmp_path / "schemas"
    written = export_schemas(out_dir=str(out_dir))

    assert set(written.keys()) == EXPECTED_TOOLS
    for tool, path in written.items():
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["tool"] == tool
        assert "input" in data and "output" in data
        assert "properties" in data["input"]


def test_load_tool_schema_roundtrip(tmp_path) -> None:
    export_schemas(out_dir=str(tmp_path))
    # load_tool_schema reads from the package SCHEMA_DIR, so export there for this check
    export_schemas()
    schema = load_tool_schema("safe_math")
    assert schema is not None
    assert "expression" in schema["input"]["properties"]


def test_export_schemas_is_idempotent(tmp_path) -> None:
    export_schemas(out_dir=str(tmp_path))
    first = {p: os.path.getmtime(os.path.join(str(tmp_path), p))
             for p in os.listdir(str(tmp_path))}
    export_schemas(out_dir=str(tmp_path))
    second = {p: os.path.getmtime(os.path.join(str(tmp_path), p))
              for p in os.listdir(str(tmp_path))}
    assert set(first) == set(second)


def test_log_tool_result_appends_and_summarizes(tmp_path) -> None:
    metrics_path = str(tmp_path / "tool_metrics.jsonl")

    ok = safe_math_tool(SafeMathInput(expression="add(1, 2)"))
    bad = safe_math_tool(SafeMathInput(expression="1/0"))  # ZeroDivisionError inside

    assert log_tool_result(ok, trace_id="trace-A", path=metrics_path)
    assert log_tool_result(bad, trace_id="trace-B", path=metrics_path)

    records = read_metrics(metrics_path)
    assert len(records) == 2
    assert records[0]["trace_id"] == "trace-A"
    assert records[0]["success"] is True
    assert records[1]["success"] is False
    assert "ZeroDivisionError" in records[1]["error"]

    summary = summarize_metrics(metrics_path)
    assert summary["safe_math"]["calls"] == 2
    assert summary["safe_math"]["success_rate"] == 0.5


def test_log_tool_result_fails_open(tmp_path) -> None:
    # Unwritable directory -> logging must return False, not raise
    ok = safe_math_tool(SafeMathInput(expression="add(1, 2)"))
    bad_path = os.path.join(str(tmp_path), "file.txt", "nested.jsonl")
    with open(os.path.join(str(tmp_path), "file.txt"), "w") as f:
        f.write("blocker")
    assert log_tool_result(ok, path=bad_path) is False
