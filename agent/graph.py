"""Autonomous Financial Reasoning Agent - Master LangGraph Orchestrator with Dynamic Fan-Out."""

from __future__ import annotations

from typing import Literal
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from agent.memory.working import get_checkpointer
from agent.nodes import (
    aggregate_sub_tasks_node,
    compute_node,
    fan_out_router,
    fuse_context_node,
    plan_node,
    read_memory_node,
    sub_task_worker,
    synthesize_node,
    write_memory_node,
)
from agent.state.schema import AgentStateV1

MAX_ITERATIONS = 3


def route_after_aggregate(state: AgentStateV1) -> Literal["compute", "fuse_context", "synthesize_answer"]:
    """Routes from parallel aggregation barrier to math compute or context fusion."""
    has_pending_math = any(
        c.get("target_tool") == "safe_math" and c.get("status") == "PENDING"
        for c in state.tool_calls
    )
    if has_pending_math:
        return "compute"

    if state.tool_results:
        return "fuse_context"

    return "synthesize_answer"


def route_after_synthesize(state: AgentStateV1) -> Literal["write_memory", "plan"]:
    """Checks termination condition: finalize if complete or loop back to plan."""
    if state.is_terminal or state.iteration_count >= MAX_ITERATIONS:
        return "write_memory"
    return "plan"


def build_financial_agent_graph(checkpointer: BaseCheckpointSaver | None = None) -> StateGraph:
    """Constructs and wires the parallelized StateGraph over AgentStateV1."""
    workflow = StateGraph(AgentStateV1)

    # 1. Register Core Nodes
    workflow.add_node("read_memory", read_memory_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("sub_task_worker", sub_task_worker)
    workflow.add_node("aggregate_sub_tasks", aggregate_sub_tasks_node)
    workflow.add_node("fuse_context", fuse_context_node)
    workflow.add_node("compute", compute_node)
    workflow.add_node("synthesize_answer", synthesize_node)
    workflow.add_node("write_memory", write_memory_node)

    # 2. Wire Control Edges
    workflow.add_edge(START, "read_memory")
    workflow.add_edge("read_memory", "plan")

    # Dynamic Fan-Out Conditional Edge using Send API
    workflow.add_conditional_edges(
        "plan",
        fan_out_router,
        ["sub_task_worker", "compute", "fuse_context", "synthesize_answer"],
    )

    # Fan-In: All parallel worker instances feed into aggregation barrier
    workflow.add_edge("sub_task_worker", "aggregate_sub_tasks")

    # Post-Aggregation Routing
    workflow.add_conditional_edges(
        "aggregate_sub_tasks",
        route_after_aggregate,
        {
            "compute": "compute",
            "fuse_context": "fuse_context",
            "synthesize_answer": "synthesize_answer",
        },
    )

    workflow.add_edge("fuse_context", "synthesize_answer")
    workflow.add_edge("compute", "synthesize_answer")

    # Verification loop from synthesis
    workflow.add_conditional_edges(
        "synthesize_answer",
        route_after_synthesize,
        {
            "write_memory": "write_memory",
            "plan": "plan",
        },
    )

    workflow.add_edge("write_memory", END)

    return workflow


def create_financial_agent(checkpointer: BaseCheckpointSaver | None = None) -> Any:
    """Compiles the LangGraph StateGraph with durable checkpointer."""
    saver = checkpointer if checkpointer is not None else get_checkpointer()
    graph = build_financial_agent_graph(checkpointer=saver)
    return graph.compile(checkpointer=saver)