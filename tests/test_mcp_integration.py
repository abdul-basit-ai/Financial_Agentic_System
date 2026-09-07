"""Validation suite for Phase 10 Model Context Protocol (MCP) Integration.

Verifies:
1. Dynamic Tool Discovery: Confirms that all core tools are published with valid JSON Schemas.
2. Resource Registration: Verifies schema inspection and prompt template resources over URIs.
3. Prompt Recipes: Verifies server-side prompt template construction.
4. Tool Execution via MCP Protocol: Invokes safe_math, code_interpreter, and graph_retrieval.
5. Client Bridge: Verifies in-process client adapter execution and schema resolution.
"""

from __future__ import annotations

import json
import pytest

from agent.mcp.client_bridge import MCPClientBridge
from agent.mcp.server import mcp


@pytest.fixture
def bridge() -> MCPClientBridge:
    return MCPClientBridge(server_instance=mcp)


# =====================================================================
# 1. Discovery & Protocol Compliance Tests
# =====================================================================


def test_mcp_tools_discovery(bridge: MCPClientBridge) -> None:
    tools = bridge.list_tools()
    tool_names = {t.name for t in tools}

    expected_tools = {
        "safe_math",
        "graph_retrieval",
        "vector_retrieval",
        "code_interpreter",
        "context_fusion",
        "memory_retrieval",
    }

    assert expected_tools.issubset(tool_names), f"Missing MCP tools: {expected_tools - tool_names}"

    # Verify input schema presence
    for t in tools:
        assert isinstance(t.input_schema, dict)
        assert len(t.description) > 10


def test_mcp_resource_read_schema(bridge: MCPClientBridge) -> None:
    raw_schema = bridge.read_resource("financial://schema/neo4j")
    schema_data = json.loads(raw_schema)

    assert "nodes" in schema_data
    assert "relationships" in schema_data
    assert "Company" in schema_data["nodes"]
    assert "(Company)-[:FILED]->(Report)" in schema_data["relationships"]


def test_mcp_resource_read_prompt(bridge: MCPClientBridge) -> None:
    raw_prompt = bridge.read_resource("financial://prompts/planner_v1")
    prompt_data = json.loads(raw_prompt)

    assert "template" in prompt_data
    assert "hash" in prompt_data
    assert "Financial Planning Agent" in prompt_data["template"]


# =====================================================================
# 2. Tool Execution via MCP Client Bridge
# =====================================================================


def test_mcp_safe_math_execution(bridge: MCPClientBridge) -> None:
    args = {"expression": "divide(subtract(1500, 1000), 1000)"}
    result = bridge.call_tool("safe_math", args)

    assert "result" in result
    assert result["result"] == 0.5
    assert result["formatted"] == "0.5"


def test_mcp_code_interpreter_execution(bridge: MCPClientBridge) -> None:
    code = """
import json

revenues = input_data.get("revs", [])
diff = revenues[1] - revenues[0]
growth = (diff / revenues[0]) * 100.0

print(json.dumps({"growth_pct": round(growth, 2)}))
"""
    args = {
        "code": code,
        "input_variables": {"revs": [120000.0, 150000.0]},
        "timeout_seconds": 5.0,
    }
    result = bridge.call_tool("code_interpreter", args)

    assert "parsed_result" in result
    assert result["parsed_result"]["growth_pct"] == 25.0
    assert result["exit_code"] == 0


def test_mcp_context_fusion_execution(bridge: MCPClientBridge) -> None:
    args = {
        "graph_records": [
            {
                "company": "MSFT",
                "report_id": "r1",
                "row_label": "Revenue",
                "category": "revenue",
                "year": 2021,
                "amount": 168000.0,
                "normalized_amount": 168000000000.0,
            }
        ],
        "vector_chunks": [
            {
                "record_id": "r1",
                "filename": "msft.txt",
                "section": "pre_text",
                "chunk_index": 0,
                "text_content": "Commercial cloud growth drove overall margin expansion.",
                "similarity_score": 0.88,
            }
        ],
        "max_context_chars": 2000,
    }
    result = bridge.call_tool("context_fusion", args)

    assert "fused_items" in result
    assert len(result["fused_items"]) == 2
    assert "formatted_context" in result
    assert "Commercial cloud" in result["formatted_context"]