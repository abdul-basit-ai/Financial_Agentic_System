"""Autonomous Financial Reasoning Agent - Model Context Protocol (MCP) Server.

Exposes tools, resources, and prompt templates over stdio / SSE transports.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from agent.sandbox.protocol import SandboxExecutionInput
from agent.tools import (
    ContextFusionInput,
    GraphMetricRecord,
    GraphQueryInput,
    MemoryQueryInput,
    SafeMathInput,
    VectorChunkRecord,
    VectorSearchInput,
    code_sandbox_tool,
    context_fusion_tool,
    graph_retrieval_tool,
    memory_retrieval_tool,
    safe_math_tool,
    vector_retrieval_tool,
)
from agent.mcp.resources import (
    get_filing_reference_resource,
    get_graph_schema_resource,
    get_prompt_template_resource,
)

# Initialize Standalone FastMCP Server
mcp = FastMCP(
    name="FinancialReasoningServer",
    instructions="Production MCP Server exposing financial knowledge graphs, vector retrieval, deterministic math, and sandboxed code execution.",
)


# =====================================================================
# 1. MCP Tools (Dynamic Model-Controlled Actions)
# =====================================================================


@mcp.tool(
    name="safe_math",
    description="Deterministically evaluates financial arithmetic and FinQA DSL formulas (add, subtract, divide, multiply, exp, table_sum, table_average) via an Abstract Syntax Tree.",
)
def tool_safe_math(expression: str) -> str:
    payload = SafeMathInput(expression=expression)
    result = safe_math_tool(payload)
    if result.success and result.data:
        return json.dumps({
            "result": result.data.result,
            "formatted": result.data.formatted,
            "expression": result.data.expression,
        })
    return json.dumps({"error": result.error})


@mcp.tool(
    name="graph_retrieval",
    description="Queries the Neo4j financial knowledge graph for exact quantitative figures (revenue, net income, operating margin) by company ticker and fiscal year.",
)
def tool_graph_retrieval(
    company_identifier: str,
    metric_name: str | None = None,
    year: int | None = None,
    record_id: str | None = None,
) -> str:
    payload = GraphQueryInput(
        company_identifier=company_identifier,
        metric_name=metric_name,
        year=year,
        record_id=record_id,
    )
    result = graph_retrieval_tool(payload)
    if result.success and result.data:
        return json.dumps(result.data.model_dump())
    return json.dumps({"error": result.error})


@mcp.tool(
    name="vector_retrieval",
    description="Performs dense semantic vector search over SEC filing narrative text chunks (MD&A commentary, risk disclosures, footnote explanations) in pgvector.",
)
def tool_vector_retrieval(
    query_text: str,
    top_k: int = 5,
    record_id: str | None = None,
    split: str | None = None,
) -> str:
    payload = VectorSearchInput(
        query_text=query_text,
        top_k=top_k,
        record_id=record_id,
        split=split,
    )
    result = vector_retrieval_tool(payload)
    if result.success and result.data:
        return json.dumps(result.data.model_dump())
    return json.dumps({"error": result.error})


@mcp.tool(
    name="code_interpreter",
    description="Executes arbitrary Python data science and modeling scripts (pandas, numpy, scipy) inside a hardened, air-gapped sandbox.",
)
def tool_code_interpreter(
    code: str,
    input_variables: dict[str, Any] | None = None,
    timeout_seconds: float = 10.0,
    memory_limit_mb: int = 512,
) -> str:
    payload = SandboxExecutionInput(
        code=code,
        input_variables=input_variables or {},
        timeout_seconds=timeout_seconds,
        memory_limit_mb=memory_limit_mb,
    )
    result = code_sandbox_tool(payload)
    if result.success and result.data:
        return json.dumps(result.data.model_dump())
    return json.dumps({"error": result.error})


@mcp.tool(
    name="context_fusion",
    description="Applies Reciprocal Rank Fusion (RRF, k=60) across heterogeneous graph line items and vector chunks to generate a deduplicated, token-budgeted context block.",
)
def tool_context_fusion(
    graph_records: list[dict[str, Any]],
    vector_chunks: list[dict[str, Any]],
    max_context_chars: int = 4000,
) -> str:
    typed_graph = [GraphMetricRecord(**r) for r in graph_records]
    typed_vector = [VectorChunkRecord(**c) for c in vector_chunks]
    payload = ContextFusionInput(
        graph_records=typed_graph,
        vector_chunks=typed_vector,
        max_context_chars=max_context_chars,
    )
    result = context_fusion_tool(payload)
    if result.success and result.data:
        return json.dumps(result.data.model_dump())
    return json.dumps({"error": result.error})


@mcp.tool(
    name="memory_retrieval",
    description="Queries historical episodic interaction summaries and procedural trajectory archetypes using multi-factor exponential decay scoring.",
)
def tool_memory_retrieval(
    query_text: str,
    company_identifier: str | None = None,
    include_procedural: bool = True,
    top_k: int = 3,
) -> str:
    payload = MemoryQueryInput(
        query_text=query_text,
        company_identifier=company_identifier,
        include_procedural=include_procedural,
        top_k=top_k,
    )
    result = memory_retrieval_tool(payload)
    if result.success and result.data:
        return json.dumps(result.data.model_dump())
    return json.dumps({"error": result.error})


# =====================================================================
# 2. MCP Resources (Passive Context Streams)
# =====================================================================


@mcp.resource("financial://schema/neo4j")
def resource_neo4j_schema() -> str:
    """Returns the structural graph schema and relationship definitions."""
    return get_graph_schema_resource()


@mcp.resource("financial://prompts/{prompt_name}")
def resource_prompt_template(prompt_name: str) -> str:
    """Returns server-managed prompt templates by name (e.g. planner_v1, synthesizer_v1)."""
    return get_prompt_template_resource(prompt_name)


@mcp.resource("financial://filings/{record_id}")
def resource_filing_reference(record_id: str) -> str:
    """Returns metadata references for a specific SEC 10-K filing container."""
    return get_filing_reference_resource(record_id)


# =====================================================================
# 3. MCP Prompts (Server-Controlled Behavioral Recipes)
# =====================================================================


@mcp.prompt(name="financial_analysis_plan")
def prompt_financial_analysis_plan(company: str, target_metric: str, periods: str) -> str:
    """Generates a structured decomposition prompt for comparative financial analysis."""
    return (
        f"You are conducting a formal financial analysis for {company}.\n"
        f"Primary Target Metric: {target_metric}\n"
        f"Target Periods: {periods}\n\n"
        "Execution Steps:\n"
        f"1. Query the knowledge graph for {target_metric} across all target periods ({periods}).\n"
        "2. Query the vector store for management discussion (MD&A) explaining changes across these periods.\n"
        "3. Compute YoY growth rates, differences, or margin ratios using safe_math or code_interpreter.\n"
        "4. Synthesize your answer with exact filing numerical citations."
    )


@mcp.prompt(name="yoy_reconciliation")
def prompt_yoy_reconciliation(company: str, year_a: int, year_b: int) -> str:
    """Generates a recipe to reconcile and compute Year-over-Year shifts."""
    return (
        f"Reconcile Year-over-Year performance for {company} between {year_a} and {year_b}.\n"
        f"1. Retrieve Base Period ({year_a}) and Comparison Period ({year_b}) line items via graph_retrieval.\n"
        f"2. Formulate the exact formula: divide(subtract(value_{year_b}, value_{year_a}), value_{year_a}).\n"
        "3. Evaluate using safe_math and report the percentage variance."
    )


if __name__ == "__main__":
    # Runs the MCP server via stdio transport when executed directly
    mcp.run(transport="stdio")