"""Nodes for structured graph retrieval, semantic vector search, and context fusion."""

from __future__ import annotations

from typing import Any

from agent.state.schema import AgentStateV1
from agent.tools.fusion_tool import ContextFusionInput, context_fusion_tool
from agent.tools.graph_tool import (
    GraphMetricRecord,
    GraphQueryInput,
    graph_retrieval_tool,
)
from agent.tools.vector_tool import (
    VectorChunkRecord,
    VectorSearchInput,
    vector_retrieval_tool,
)


def retrieve_graph_node(state: AgentStateV1) -> dict[str, Any]:
    """Executes pending graph retrieval tool calls against Neo4j."""
    remaining_calls: list[dict[str, Any]] = []
    executed_results: list[dict[str, Any]] = []
    scratchpad_logs: list[str] = []

    for call in state.tool_calls:
        if call.get("target_tool") == "graph_retrieval" and call.get("status") == "PENDING":
            payload = call.get("payload", {})
            query_input = GraphQueryInput(
                company_identifier=payload.get("company_identifier", state.company_identifier or "UNKNOWN"),
                metric_name=payload.get("metric_name"),
                year=payload.get("year"),
                record_id=payload.get("record_id"),
            )
            result = graph_retrieval_tool(query_input)
            executed_results.append({
                "task_id": call.get("task_id"),
                "tool_name": "graph_retrieval",
                "success": result.success,
                "data": result.data.model_dump() if result.data else None,
                "error": result.error,
            })
            count = result.data.total_found if result.data else 0
            scratchpad_logs.append(f"[Graph Retrieval] Fetched {count} records for year={payload.get('year')}, metric={payload.get('metric_name')}.")
        else:
            remaining_calls.append(dict(call))
    # Executed calls leave the queue; no need to re-add their updated copies.
    # (Their completion is recorded in tool_results; re-adding would double-count.)

    return {
        "tool_calls": remaining_calls,
        "tool_results": executed_results,
        "scratchpad": scratchpad_logs,
    }


def retrieve_vector_node(state: AgentStateV1) -> dict[str, Any]:
    """Executes pending vector retrieval tool calls against pgvector."""
    remaining_calls: list[dict[str, Any]] = []
    executed_results: list[dict[str, Any]] = []
    scratchpad_logs: list[str] = []

    for call in state.tool_calls:
        if call.get("target_tool") == "vector_retrieval" and call.get("status") == "PENDING":
            payload = call.get("payload", {})
            query_input = VectorSearchInput(
                query_text=payload.get("query_text", state.input),
                top_k=payload.get("top_k", 5),
            )
            result = vector_retrieval_tool(query_input)
            executed_results.append({
                "task_id": call.get("task_id"),
                "tool_name": "vector_retrieval",
                "success": result.success,
                "data": result.data.model_dump() if result.data else None,
                "error": result.error,
            })
            count = result.data.total_found if result.data else 0
            scratchpad_logs.append(f"[Vector Retrieval] Fetched {count} narrative chunks for query: '{payload.get('query_text')}'.")
        else:
            remaining_calls.append(dict(call))

    return {
        "tool_calls": remaining_calls,
        "tool_results": executed_results,
        "scratchpad": scratchpad_logs,
    }


def fuse_context_node(state: AgentStateV1) -> dict[str, Any]:
    """Fuses multi-modal graph records and text chunks using Reciprocal Rank Fusion.

    Persists the fused context into scratchpad so the synthesizer (and Phase 8
    HITL reviewers) can see exactly which evidence blocks were selected.
    """
    graph_records: list[GraphMetricRecord] = []
    vector_chunks: list[VectorChunkRecord] = []

    for res in state.tool_results:
        data = res.get("data")
        if not data:
            continue
        if res.get("tool_name") == "graph_retrieval" and "records" in data:
            for r in data["records"]:
                graph_records.append(GraphMetricRecord(**r))
        elif res.get("tool_name") == "vector_retrieval" and "chunks" in data:
            for c in data["chunks"]:
                vector_chunks.append(VectorChunkRecord(**c))

    fusion_input = ContextFusionInput(
        graph_records=graph_records,
        vector_chunks=vector_chunks,
        max_context_chars=3500,
    )
    fusion_result = context_fusion_tool(fusion_input)

    fused_preview = ""
    if fusion_result.data and fusion_result.data.formatted_context:
        fused_preview = fusion_result.data.formatted_context[:600]

    log_entry = (
        f"[Context Fusion] RRF fused {fusion_result.data.total_input_items if fusion_result.data else 0} "
        f"items into {fusion_result.data.retained_items if fusion_result.data else 0} token-budgeted blocks."
    )

    return {
        "scratchpad": [log_entry, f"[Fused Context Preview]\n{fused_preview}"],
    }
