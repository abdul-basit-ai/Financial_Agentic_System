"""Master planning node that analyzes intent and emits structured tool calls."""

from __future__ import annotations

import os
from typing import Any

from agent.prompts import load_prompt
from agent.state.schema import AgentStateV1
from agent.tools.decomposer import decompose_query

GRAPH_VERSION = "v1.0.0"


def plan_node(state: AgentStateV1) -> dict[str, Any]:
    """Evaluates question and scratchpad to construct or refine tool execution queue."""
    _, prompt_version, prompt_hash = load_prompt("planner_v1")
    new_iteration = state.iteration_count + 1

    # Check if this is an iterative refinement loop
    if state.tool_results and not state.final_answer:
        log_entry = (
            f"[Planner {prompt_version}@{GRAPH_VERSION} (hash:{prompt_hash})] "
            f"Iteration {new_iteration}: Reviewing {len(state.tool_results)} completed tool results."
        )
        return {
            "iteration_count": new_iteration,
            "scratchpad": [log_entry],
            "risk_evaluated": False,
        }

    # Macro-decomposition of user financial question
    company = state.company_identifier or "UNKNOWN"
    decomp = decompose_query(query=state.input, company=company)

    pending_calls: list[dict[str, Any]] = []
    for task in decomp.sub_tasks:
        pending_calls.append({
            "task_id": task.task_id,
            "target_tool": task.target_tool,
            "payload": task.query_payload,
            "dependencies": task.dependencies,
            "status": "PENDING",
        })

    log_entry = (
        f"[Planner {prompt_version}@{GRAPH_VERSION} (hash:{prompt_hash})] "
        f"Generated plan with {len(pending_calls)} tasks. Strategy: {decomp.reasoning_plan}"
    )

    return {
        "iteration_count": new_iteration,
        "tool_calls": pending_calls,
        "scratchpad": [log_entry],
        "risk_evaluated": False,
    }