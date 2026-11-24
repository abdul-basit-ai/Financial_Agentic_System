"""Use run-scoped working memory for financial answers."""

from __future__ import annotations

from typing import Any

from agent.state.schema import AgentStateV1


def read_memory_node(state: AgentStateV1) -> dict[str, Any]:
    return {"memory_refs": [], "scratchpad": ["[Memory] Using run-scoped evidence."]}


def write_memory_node(state: AgentStateV1) -> dict[str, Any]:
    return {"is_terminal": True, "scratchpad": ["[Memory Promotion] Disabled."]}
