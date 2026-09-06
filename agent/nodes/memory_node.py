"""Graph nodes for memory tier reads and writes."""

from __future__ import annotations

from typing import Any

from agent.memory.promoter import MemoryPromotionEngine
from agent.state.schema import AgentStateV1
from agent.tools.memory_tool import MemoryQueryInput, memory_retrieval_tool


def read_memory_node(state: AgentStateV1) -> dict[str, Any]:
    """Retrieves relevant episodic memories and procedural archetypes at graph entry."""
    query = state.input
    company = state.company_identifier

    # Retrieve context from episodic & procedural memory
    mem_input = MemoryQueryInput(
        query_text=query,
        company_identifier=company,
        include_procedural=True,
        top_k=2,
    )
    result = memory_retrieval_tool(mem_input)

    scratchpad_entries = ["[Memory] Initialized working session."]
    memory_refs: list[dict[str, Any]] = []

    if result.success and result.data:
        for ep in result.data.episodic_memories:
            memory_refs.append({"type": "episodic", "id": ep["id"], "summary": ep["summary"]})
            scratchpad_entries.append(f"[Memory - Past Episode] {ep['summary']}")

        if result.data.matched_archetype:
            arch = result.data.matched_archetype
            memory_refs.append({"type": "procedural", "id": arch["archetype_id"]})
            scratchpad_entries.append(
                f"[Memory - Matched Plan Archetype: {arch['name']}]\nDemonstration:\n{arch['few_shot_prompt']}"
            )

    return {
        "scratchpad": scratchpad_entries,
        "memory_refs": memory_refs,
    }


def write_memory_node(state: AgentStateV1) -> dict[str, Any]:
    """Promotes completed session state into durable long-term episodic memory."""
    try:
        engine = MemoryPromotionEngine()
        session_id = f"sess_{state.trace_id[:8]}"
        promoted_id = engine.promote_session(session_id=session_id, state=state)
        status_msg = f"[Memory Promotion] Promoted episode ID #{promoted_id}" if promoted_id else "[Memory Promotion] Skipped promotion."
    except Exception as exc:
        status_msg = f"[Memory Promotion] Failed: {exc}"

    return {
        "scratchpad": [status_msg],
        "is_terminal": True,
    }