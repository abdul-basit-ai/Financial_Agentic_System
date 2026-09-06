"""Promotion engine: filters, scores, and distills working memory into episodic/semantic tiers."""

from __future__ import annotations

import math
from typing import Any

from agent.memory.episodic import EpisodicEntry, EpisodicMemoryStore
from agent.state.schema import AgentStateV1


def calculate_episode_importance(state: AgentStateV1) -> float:
    """Scores analytical importance: S_imp = sigmoid(0.3 * tools + 0.5 * hitl + 0.4 * subtasks)."""
    n_tools = len(state.tool_results)
    hitl_weight = 1.0 if state.hitl_status in {"APPROVED", "PENDING"} else 0.0
    n_subtasks = len(state.sub_task_results)

    raw_score = (0.3 * n_tools) + (0.5 * hitl_weight) + (0.4 * n_subtasks)
    return round(1.0 / (1.0 + math.exp(-raw_score)), 4)


class MemoryPromotionEngine:
    """Distills end-of-session working memory into durable long-term storage."""

    def __init__(self, episodic_store: EpisodicMemoryStore | None = None) -> None:
        self.episodic_store = episodic_store or EpisodicMemoryStore()

    def promote_session(
        self, session_id: str, state: AgentStateV1
    ) -> int | None:
        # Promotion Filter: Do not record incomplete or failed sessions
        if not state.final_answer and not state.sub_task_results:
            return None

        importance = calculate_episode_importance(state)

        # Distill key findings from state
        findings: dict[str, Any] = {
            "completed_tools": [tr.get("tool_name") for tr in state.tool_results],
            "sub_task_outputs": state.sub_task_results,
            "hitl_action": state.hitl_status,
        }

        summary = state.final_answer or (
            f"Executed {len(state.tool_results)} tool steps for query: {state.input[:100]}"
        )

        entry = EpisodicEntry(
            session_id=session_id,
            trace_id=state.trace_id,
            company_identifier=state.company_identifier,
            query_text=state.input,
            summary=summary,
            key_findings=findings,
            importance_score=importance,
        )

        return self.episodic_store.save_episode(entry)