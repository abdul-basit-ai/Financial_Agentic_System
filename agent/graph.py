"""Financial reasoning graph with a review boundary after all calculations."""

from __future__ import annotations

from typing import Any, Literal

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
from agent.nodes.parallel_nodes import WorkerPayload
from agent.nodes.synthesize_node import MAX_GRAPH_ITERATIONS, program_node
from agent.state.schema import AgentStateV1


def route_after_risk(state: AgentStateV1) -> Literal["hitl_gate", "synthesize_answer"]:
    return "hitl_gate" if state.hitl_status == "PENDING" else "synthesize_answer"


def route_after_override(
    state: AgentStateV1,
) -> Literal["fan_out_router_node", "synthesize_answer", "write_memory"]:
    if state.hitl_status in {"REJECTED", "REJECT"} or state.is_terminal:
        return "write_memory"
    if state.hitl_status == "EDIT":
        return "fan_out_router_node"
    return "synthesize_answer"


def route_after_synthesize(state: AgentStateV1) -> Literal["write_memory", "plan"]:
    if state.is_terminal or state.iteration_count >= MAX_GRAPH_ITERATIONS:
        return "write_memory"
    return "plan"


def build_financial_agent_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> StateGraph:
    workflow = StateGraph(AgentStateV1)
    workflow.add_node("read_memory", read_memory_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("sub_task_worker", sub_task_worker, input_schema=WorkerPayload)
    workflow.add_node("aggregate_sub_tasks", aggregate_sub_tasks_node)
    workflow.add_node("fan_out_router_node", fan_out_router_node)
    workflow.add_node("fuse_context", fuse_context_node)
    workflow.add_node("compute", compute_node)
    workflow.add_node("program", program_node)
    workflow.add_node("eval_risk", eval_financial_risk_node)
    workflow.add_node("hitl_gate", hitl_gate_node)
    workflow.add_node("apply_override", apply_override_node)
    workflow.add_node("synthesize_answer", synthesize_node)
    workflow.add_node("write_memory", write_memory_node)

    workflow.add_edge(START, "read_memory")
    workflow.add_edge("read_memory", "plan")
    for origin in ("plan", "aggregate_sub_tasks", "fan_out_router_node"):
        workflow.add_conditional_edges(
            origin, fan_out_router, ["sub_task_worker", "fuse_context"]
        )
    workflow.add_edge("sub_task_worker", "aggregate_sub_tasks")
    workflow.add_edge("fuse_context", "compute")
    workflow.add_edge("compute", "program")
    workflow.add_edge("program", "eval_risk")
    workflow.add_conditional_edges("eval_risk", route_after_risk)
    workflow.add_edge("hitl_gate", "apply_override")
    workflow.add_conditional_edges("apply_override", route_after_override)
    workflow.add_conditional_edges("synthesize_answer", route_after_synthesize)
    workflow.add_edge("write_memory", END)
    return workflow


def create_financial_agent(checkpointer: BaseCheckpointSaver | None = None) -> Any:
    saver = checkpointer if checkpointer is not None else get_checkpointer()
    return (
        build_financial_agent_graph()
        .compile(checkpointer=saver)
        .with_config({"recursion_limit": 150})
    )
