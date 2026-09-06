"""Autonomous Financial Reasoning Agent - Master LangGraph Orchestrator."""

from __future__ import annotations

from typing import Literal
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from agent.memory.working import get_checkpointer
from agent.nodes import (
    compute_node,
    fuse_context_node,
    plan_node,
    read_memory_node,
    retrieve_graph_node,
    retrieve_vector_node,
    synthesize_node,
    write_memory_node,
)
from agent.state.schema import AgentStateV1

MAX_ITERATIONS = 3


def route_after_plan(state: AgentStateV1) -> Literal["retrieve_graph", "retrieve_vector", "compute", "synthesize_answer"]:
    """Determines next execution node from pending tool calls."""
    pending = [c for c in state.tool_calls if c.get("status") == "PENDING"]
    if not pending:
        return "synthesize_answer"

    next_tool = pending[0].get("target_tool")
    if next_tool == "graph_retrieval":
        return "retrieve_graph"
    if next_tool == "vector_retrieval":
        return "retrieve_vector"
    if next_tool == "safe_math":
        return "compute"

    return "synthesize_answer"


def route_after_retrieval(state: AgentStateV1) -> Literal["retrieve_vector", "retrieve_graph", "fuse_context"]:
    """Routes sequential retrieval steps before triggering context fusion."""
    pending = [c for c in state.tool_calls if c.get("status") == "PENDING"]
    for c in pending:
        if c.get("target_tool") == "vector_retrieval":
            return "retrieve_vector"
        if c.get("target_tool") == "graph_retrieval":
            return "retrieve_graph"
    return "fuse_context"


def route_after_fuse(state: AgentStateV1) -> Literal["compute", "synthesize_answer"]:
    """Determines whether to execute safe arithmetic or proceed directly to synthesis."""
    pending = [c for c in state.tool_calls if c.get("status") == "PENDING"]
    for c in pending:
        if c.get("target_tool") == "safe_math":
            return "compute"
    return "synthesize_answer"


def route_after_synthesize(state: AgentStateV1) -> Literal["write_memory", "plan"]:
    """Checks termination condition: finalize if complete or loop back to plan."""
    if state.is_terminal or state.iteration_count >= MAX_ITERATIONS:
        return "write_memory"
    return "plan"


def build_financial_agent_graph(checkpointer: BaseCheckpointSaver | None = None) -> StateGraph:
    """Constructs and wires the single StateGraph over AgentStateV1."""
    workflow = StateGraph(AgentStateV1)

    # 1. Register Nodes
    workflow.add_node("read_memory", read_memory_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("retrieve_graph", retrieve_graph_node)
    workflow.add_node("retrieve_vector", retrieve_vector_node)
    workflow.add_node("fuse_context", fuse_context_node)
    workflow.add_node("compute", compute_node)
    workflow.add_node("synthesize_answer", synthesize_node)
    workflow.add_node("write_memory", write_memory_node)

    # 2. Wire Control Edges
    workflow.add_edge(START, "read_memory")
    workflow.add_edge("read_memory", "plan")

    # Conditional router from plan
    workflow.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "retrieve_graph": "retrieve_graph",
            "retrieve_vector": "retrieve_vector",
            "compute": "compute",
            "synthesize_answer": "synthesize_answer",
        },
    )

    # Routers from retrieval
    workflow.add_conditional_edges(
        "retrieve_graph",
        route_after_retrieval,
        {
            "retrieve_vector": "retrieve_vector",
            "retrieve_graph": "retrieve_graph",
            "fuse_context": "fuse_context",
        },
    )
    workflow.add_edge("retrieve_vector", "fuse_context")

    # Router from context fusion
    workflow.add_conditional_edges(
        "fuse_context",
        route_after_fuse,
        {
            "compute": "compute",
            "synthesize_answer": "synthesize_answer",
        },
    )

    # From compute to synthesis
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

    # Terminal node to end
    workflow.add_edge("write_memory", END)

    return workflow


def create_financial_agent(checkpointer: BaseCheckpointSaver | None = None) -> Any:
    """Compiles the LangGraph StateGraph with durable checkpointer."""
    saver = checkpointer if checkpointer is not None else get_checkpointer()
    graph = build_financial_agent_graph(checkpointer=saver)
    return graph.compile(checkpointer=saver)