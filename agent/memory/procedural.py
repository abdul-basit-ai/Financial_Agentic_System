"""Procedural memory store: query archetype routing and trajectory demonstration bank."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class ExecutionStep(BaseModel):
    step_index: int
    action_type: str
    target_tool: str
    rationale: str


class TrajectoryArchetype(BaseModel):
    archetype_id: str
    name: str
    description: str
    example_query: str
    planned_steps: list[ExecutionStep]
    few_shot_prompt: str


STANDARD_ARCHETYPES: list[TrajectoryArchetype] = [
    TrajectoryArchetype(
        archetype_id="yoy_delta_comparison",
        name="Year-over-Year Delta & Growth Analysis",
        description="Computes change, absolute delta, or percentage growth across two fiscal periods.",
        example_query="What was the percentage change in Apple's revenue from 2019 to 2020?",
        planned_steps=[
            ExecutionStep(step_index=1, action_type="RETRIEVE", target_tool="graph_retrieval", rationale="Lookup period 1 value"),
            ExecutionStep(step_index=2, action_type="RETRIEVE", target_tool="graph_retrieval", rationale="Lookup period 2 value"),
            ExecutionStep(step_index=3, action_type="COMPUTE", target_tool="safe_math", rationale="Calculate difference and divide by base period"),
        ],
        few_shot_prompt=(
            "Plan:\n"
            "1. Tool: graph_retrieval (company='AAPL', year=2019, metric='revenue')\n"
            "2. Tool: graph_retrieval (company='AAPL', year=2020, metric='revenue')\n"
            "3. Tool: safe_math (expression='divide(subtract(rev_2020, rev_2019), rev_2019)')"
        ),
    ),
    TrajectoryArchetype(
        archetype_id="qualitative_driver_explanation",
        name="Financial Driver & MD&A Explanation",
        description="Explains reasons, causes, or commentary regarding a line-item shift.",
        example_query="Why did operating income decrease in 2020?",
        planned_steps=[
            ExecutionStep(step_index=1, action_type="RETRIEVE_STRUCTURED", target_tool="graph_retrieval", rationale="Verify underlying operating values"),
            ExecutionStep(step_index=2, action_type="RETRIEVE_NARRATIVE", target_tool="vector_retrieval", rationale="Search MD&A text chunks for drivers"),
            ExecutionStep(step_index=3, action_type="FUSE", target_tool="context_fusion", rationale="Synthesize numbers with narrative reasoning"),
        ],
        few_shot_prompt=(
            "Plan:\n"
            "1. Tool: graph_retrieval (year=2020, metric='operating_income')\n"
            "2. Tool: vector_retrieval (query_text='operating income decrease drivers')\n"
            "3. Tool: context_fusion"
        ),
    ),
]


class ProceduralMemoryBank:
    """Matches incoming queries against verified procedural execution archetypes."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name)
        self.archetypes = {a.archetype_id: a for a in STANDARD_ARCHETYPES}
        self._embeddings: dict[str, list[float]] = {}
        self._precompute_embeddings()

    def _precompute_embeddings(self) -> None:
        for arch_id, arch in self.archetypes.items():
            corpus = f"{arch.name} {arch.description} {arch.example_query}"
            emb = self.model.encode(corpus, normalize_embeddings=True).tolist()
            self._embeddings[arch_id] = emb

    def match_archetype(
        self, query: str, threshold: float = 0.55
    ) -> tuple[TrajectoryArchetype | None, float]:
        query_emb = self.model.encode(query, normalize_embeddings=True).tolist()
        best_match = None
        best_score = -1.0

        for arch_id, arch_emb in self._embeddings.items():
            sim = sum(q * a for q, a in zip(query_emb, arch_emb))
            if sim > best_score:
                best_score = sim
                best_match = self.archetypes[arch_id]

        if best_match and best_score >= threshold:
            return best_match, round(best_score, 4)
        return None, round(best_score, 4)