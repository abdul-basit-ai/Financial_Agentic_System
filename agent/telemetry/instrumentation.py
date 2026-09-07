"""Tracing instrumentation hooks for database and tool executions."""

from __future__ import annotations

from typing import Any

from agent.telemetry.tracer import trace_operation


def instrument_graph_query(company: str, year: int | None, metric: str | None) -> Any:
    """Constructs a tracing wrapper for Neo4j queries."""
    return trace_operation(
        name="neo4j_cypher_query",
        run_type="retriever",
        metadata={"company": company, "year": year, "metric": metric},
    )


def instrument_vector_search(query: str, top_k: int) -> Any:
    """Constructs a tracing wrapper for pgvector searches."""
    return trace_operation(
        name="pgvector_search",
        run_type="retriever",
        metadata={"query_length": len(query), "top_k": top_k},
    )


def instrument_sandbox_execution(timeout: float, memory_mb: int) -> Any:
    """Constructs a tracing wrapper for Python sandbox execution."""
    return trace_operation(
        name="sandbox_python_eval",
        run_type="tool",
        metadata={"timeout_s": timeout, "memory_mb": memory_mb},
    )