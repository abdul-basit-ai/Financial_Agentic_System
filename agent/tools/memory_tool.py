"""First-class memory retrieval tool integrating episodic and procedural tiers."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

from agent.memory.episodic import EpisodicMemoryStore, EpisodicRetrievalResult
from agent.memory.procedural import ProceduralMemoryBank, TrajectoryArchetype
from agent.tools.base import ToolResult


class MemoryQueryInput(BaseModel):
    query_text: str = Field(..., description="Semantic query to match past episodes or patterns")
    company_identifier: str | None = Field(
        default=None, description="Company filter for episodic memory"
    )
    include_procedural: bool = Field(
        default=True, description="Whether to include matching execution archetypes"
    )
    top_k: int = Field(default=3, ge=1, le=10, description="Max episodic memories to return")


class MemoryQueryOutput(BaseModel):
    episodic_memories: list[dict[str, Any]]
    matched_archetype: dict[str, Any] | None
    total_found: int


class MemoryRetrievalTool:
    """Orchestrates multi-tier memory queries."""

    def __init__(
        self,
        episodic_store: EpisodicMemoryStore | None = None,
        procedural_bank: ProceduralMemoryBank | None = None,
    ) -> None:
        self.episodic_store = episodic_store or EpisodicMemoryStore()
        self.procedural_bank = procedural_bank or ProceduralMemoryBank()

    def search(
        self,
        query_text: str,
        company_identifier: str | None = None,
        include_procedural: bool = True,
        top_k: int = 3,
    ) -> MemoryQueryOutput:
        episodes: list[EpisodicRetrievalResult] = []
        try:
            episodes = self.episodic_store.retrieve_episodes(
                query=query_text,
                company_identifier=company_identifier,
                top_k=top_k,
            )
        except Exception as exc:
            # Degrade gracefully to zero episodic context, but surface the reason
            # (Phase 12 tracing needs this; silent swallowing hides config errors).
            print(f"[memory_tool] episodic retrieval degraded: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            episodes = []

        formatted_episodes = [
            {
                "id": ep.id,
                "session_id": ep.session_id,
                "query": ep.query_text,
                "summary": ep.summary,
                "final_score": ep.final_score,
                "created_at_recency": ep.recency_score,
            }
            for ep in episodes
        ]

        matched_archetype_dict = None
        if include_procedural:
            arch, score = self.procedural_bank.match_archetype(query_text)
            if arch:
                matched_archetype_dict = {
                    "archetype_id": arch.archetype_id,
                    "name": arch.name,
                    "similarity": score,
                    "few_shot_prompt": arch.few_shot_prompt,
                }

        return MemoryQueryOutput(
            episodic_memories=formatted_episodes,
            matched_archetype=matched_archetype_dict,
            total_found=len(formatted_episodes) + (1 if matched_archetype_dict else 0),
        )


def memory_retrieval_tool(
    payload: MemoryQueryInput, client: MemoryRetrievalTool | None = None
) -> ToolResult[MemoryQueryOutput]:
    """Instrumented tool entrypoint for memory retrieval."""
    instance = client or MemoryRetrievalTool()
    return ToolResult.execute_instrumented(
        tool_name="memory_retrieval",
        fn=instance.search,
        query_text=payload.query_text,
        company_identifier=payload.company_identifier,
        include_procedural=payload.include_procedural,
        top_k=payload.top_k,
    )