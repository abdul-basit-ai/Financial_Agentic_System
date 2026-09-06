"""Autonomous Financial Reasoning Agent - Master LangGraph Orchestrator with HITL Governance."""

from __future__ import annotations

from typing import Literal
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from agent.memory.working import get_checkpointer
from agent.nodes import (
    aggregate_sub_tasks_node,
    apply_override_node,
    compute_node,
    eval_financial_risk_node,
    fan_out_router,
    fan_out_router_node,
    fuse_context_node,
    hitl_gate_node,
    plan_node,
    read_memory_node,
    sub_task_worker,
    synthesize_node,
    write_memory_node,
)
from agent.state.schema import AgentStateV1

MAX_ITERATIONS = 3


def route_after_aggregate(state: AgentStateV1) -> Literal["compute", "fuse_context", "eval_risk", "synthesize_answer"]:
    """Routes from parallel aggregation to risk evaluation or context fusion."""
    if state.tool_results:
        return "fuse_context"
    return "eval_risk"


def route_after_risk(state: AgentStateV1) -> Literal["hitl_gate", "compute", "synthesize_answer", "fan_out"]:
    """Conditional router checking if risk triggers require human interrupt."""
    if state.hitl_status == "PENDING":
        return "hitl_gate"

    # After a reviewer EDIT re-enters governance, pending edited retrieval
    # calls must still execute — route back to the fan-out.
    pending_retrieval = any(
        c.get("target_tool") in {"graph_retrieval", "vector_retrieval"}
        and c.get("status") == "PENDING"
        for c in state.tool_calls
    )
    if pending_retrieval and state.risk_evaluated and state.hitl_status == "APPROVED":
        return "fan_out"

    has_pending_math = any(
        c.get("target_tool") == "safe_math" and c.get("status") == "PENDING"
        for c in state.tool_calls
    )
    if has_pending_math:
        return "compute"

    return "synthesize_answer"


def route_after_compute(state: AgentStateV1) -> Literal["eval_risk", "synthesize_answer"]:
    """Computed figures are exactly the kind of output governance must review
    (extreme growth, impossible ratios). Route post-math through risk eval —
    but only once, by tracking that eval already ran this iteration."""
    if not state.risk_evaluated:
        return "eval_risk"
    return "synthesize_answer"


def route_after_override(state: AgentStateV1) -> Literal["fan_out", "compute", "synthesize_answer", "write_memory"]:
    """Routes after human intervention is applied.

    EDIT: dispatch the reviewer's edited calls FIRST (fan-out), then the
    normal aggregate -> fuse -> eval_risk cycle gates on the fresh results.
    Routing to eval_risk here would re-pause the gate before execution.
    """
    if state.is_terminal:
        return "write_memory"

    if state.hitl_status in {"REJECTED", "REJECT"}:
        return "write_memory"

    # EDIT: reviewer replaced tool_calls — dispatch them under governance.
    if state.hitl_status == "EDIT":
        pending_exec = [
            c for c in state.tool_calls
            if c.get("target_tool") in {"graph_retrieval", "vector_retrieval"}
            and c.get("status") == "PENDING"
        ]
        if pending_exec:
            return "fan_out"
        if not state.final_answer:
            return "synthesize_answer"

    has_pending_math = any(
        c.get("target_tool") == "safe_math" and c.get("status") == "PENDING"
        for c in state.tool_calls
    )
    if has_pending_math:
        return "compute"

    if not state.final_answer:
        return "synthesize_answer"

    return "write_memory"


def route_after_synthesize(state: AgentStateV1) -> Literal["write_memory", "plan"]:
    """Checks termination condition: finalize if complete or loop back to plan."""
    if state.is_terminal or state.iteration_count >= MAX_ITERATIONS:
        return "write_memory"
    return "plan"


def build_financial_agent_graph(checkpointer: BaseCheckpointSaver | None = None) -> StateGraph:
    """Constructs and wires the parallelized StateGraph with Phase 8 HITL governance."""
    workflow = StateGraph(AgentStateV1)

    # 1. Register All Nodes
    workflow.add_node("read_memory", read_memory_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("sub_task_worker", sub_task_worker)
    workflow.add_node("aggregate_sub_tasks", aggregate_sub_tasks_node)
    # Plain-node wrapper so EDIT re-entry can route into the Send fan-out
    # (a conditional edge cannot target another conditional edge).
    workflow.add_node("fan_out_router_node", fan_out_router_node)
    workflow.add_node("fuse_context", fuse_context_node)
    workflow.add_node("eval_risk", eval_financial_risk_node)
    workflow.add_node("hitl_gate", hitl_gate_node)
    workflow.add_node("apply_override", apply_override_node)
    workflow.add_node("compute", compute_node)
    workflow.add_node("synthesize_answer", synthesize_node)
    workflow.add_node("write_memory", write_memory_node)

    # 2. Wire Graph Control Edges
    workflow.add_edge(START, "read_memory")
    workflow.add_edge("read_memory", "plan")

    # Dynamic Fan-Out from Planner
    workflow.add_conditional_edges(
        "plan",
        fan_out_router,
        ["sub_task_worker", "compute", "fuse_context", "synthesize_answer"],
    )

    # Fan-out wrapper node -> same Send dispatch (EDIT re-entry path)
    workflow.add_conditional_edges(
        "fan_out_router_node",
        fan_out_router,
        ["sub_task_worker", "compute", "fuse_context", "synthesize_answer"],
    )

    # Fan-In to Aggregator
    workflow.add_edge("sub_task_worker", "aggregate_sub_tasks")

    # Post-Aggregation Routing
    workflow.add_conditional_edges(
        "aggregate_sub_tasks",
        route_after_aggregate,
        {
            "compute": "compute",
            "fuse_context": "fuse_context",
            "eval_risk": "eval_risk",
            "synthesize_answer": "synthesize_answer",
        },
    )

    # Post-Fusion leads directly into Risk Evaluation
    workflow.add_edge("fuse_context", "eval_risk")

    # Risk Evaluation Routing (HITL Gate vs Autonomous vs EDIT re-execution)
    workflow.add_conditional_edges(
        "eval_risk",
        route_after_risk,
        {
            "hitl_gate": "hitl_gate",
            "fan_out": "fan_out_router_node",
            "compute": "compute",
            "synthesize_answer": "synthesize_answer",
        },
    )

    # HITL Gate leads to Apply Override
    workflow.add_edge("hitl_gate", "apply_override")

    # Post-Override Routing (EDIT re-enters governance loop with edited calls)
    workflow.add_conditional_edges(
        "apply_override",
        route_after_override,
        {
            "fan_out": "fan_out_router_node",
            "compute": "compute",
            "synthesize_answer": "synthesize_answer",
            "write_memory": "write_memory",
        },
    )

    # Computed figures must pass governance before synthesis (one eval per
    # iteration, enforced by the risk_evaluated flag)
    workflow.add_conditional_edges(
        "compute",
        route_after_compute,
        {
            "eval_risk": "eval_risk",
            "synthesize_answer": "synthesize_answer",
        },
    )

    # Final Verification Routing
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