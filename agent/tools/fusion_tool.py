"""Context fusion tool implementing Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

from agent.tools.base import ToolResult
from agent.tools.graph_tool import GraphMetricRecord
from agent.tools.vector_tool import VectorChunkRecord

RRF_DEFAULT_K = 60


class ContextFusionInput(BaseModel):
    graph_records: list[GraphMetricRecord] = Field(default_factory=list)
    vector_chunks: list[VectorChunkRecord] = Field(default_factory=list)
    graph_weight: float = Field(default=1.2, ge=0.1, le=5.0)
    vector_weight: float = Field(default=1.0, ge=0.1, le=5.0)
    max_context_chars: int = Field(default=4000, ge=500, le=16000)


class FusedItem(BaseModel):
    item_id: str
    source: str
    content: str
    rrf_score: float
    metadata: dict[str, Any]


class ContextFusionOutput(BaseModel):
    fused_items: list[FusedItem]
    formatted_context: str
    total_input_items: int
    retained_items: int


def reciprocal_rank_fusion(
    graph_records: list[GraphMetricRecord],
    vector_chunks: list[VectorChunkRecord],
    graph_weight: float = 1.2,
    vector_weight: float = 1.0,
    max_context_chars: int = 4000,
    k: int = RRF_DEFAULT_K,
) -> ContextFusionOutput:
    scores: dict[str, float] = {}
    item_lookup: dict[str, FusedItem] = {}

    # 1. Rank Graph Records (with None guards)
    for rank, g in enumerate(graph_records, start=1):
        item_id = f"graph::{g.report_id}::{g.row_label}::{g.year}"
        amount_str = f"{g.amount:,.2f}" if g.amount is not None else "N/A"
        base_str = f"{g.normalized_amount:,.2f}" if g.normalized_amount is not None else "N/A"

        content = (
            f"[Table Line] Company: {g.company} | Year: {g.year or 'N/A'} | "
            f"{g.row_label} = {amount_str} (Base: {base_str})"
        )
        rrf_contribution = graph_weight / (k + rank)
        scores[item_id] = scores.get(item_id, 0.0) + rrf_contribution
        item_lookup[item_id] = FusedItem(
            item_id=item_id,
            source="graph",
            content=content,
            rrf_score=0.0,
            metadata={"category": g.category, "year": g.year},
        )

    # 2. Rank Vector Chunks
    for rank, v in enumerate(vector_chunks, start=1):
        item_id = f"vector::{v.record_id}::{v.section}::{v.chunk_index}"
        content = f"[Narrative - {v.section}] {v.text_content}"
        rrf_contribution = vector_weight / (k + rank)
        scores[item_id] = scores.get(item_id, 0.0) + rrf_contribution
        item_lookup[item_id] = FusedItem(
            item_id=item_id,
            source="vector",
            content=content,
            rrf_score=0.0,
            metadata={"similarity": v.similarity_score, "record_id": v.record_id},
        )

    # 3. Sort by fused score
    sorted_item_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused_items: list[FusedItem] = []
    formatted_blocks: list[str] = []
    current_char_count = 0

    for i_id in sorted_item_ids:
        item = item_lookup[i_id]
        item.rrf_score = round(scores[i_id], 6)
        item_block = f"{item.content}\n"

        if current_char_count + len(item_block) > max_context_chars and fused_items:
            break

        fused_items.append(item)
        formatted_blocks.append(item_block)
        current_char_count += len(item_block)

    formatted_context = "\n".join(formatted_blocks).strip()

    return ContextFusionOutput(
        fused_items=fused_items,
        formatted_context=formatted_context,
        total_input_items=len(graph_records) + len(vector_chunks),
        retained_items=len(fused_items),
    )


def context_fusion_tool(payload: ContextFusionInput) -> ToolResult[ContextFusionOutput]:
    """Instrumented tool entrypoint for context fusion."""
    return ToolResult.execute_instrumented(
        tool_name="context_fusion",
        fn=reciprocal_rank_fusion,
        graph_records=payload.graph_records,
        vector_chunks=payload.vector_chunks,
        graph_weight=payload.graph_weight,
        vector_weight=payload.vector_weight,
        max_context_chars=payload.max_context_chars,
    )