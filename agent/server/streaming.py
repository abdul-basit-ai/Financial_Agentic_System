"""Server-Sent Events (SSE) streaming engine for LangGraph execution."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from agent.server.schemas import SSEEvent
from agent.state.schema import AgentStateV1


async def stream_agent_events(
    agent: Any,
    initial_state: AgentStateV1,
    config: dict[str, Any],
) -> AsyncGenerator[str, None]:
    """Streams LangGraph execution events as SSE packets.

    Emits:
      - 'lifecycle': Start/initialization metadata.
      - 'node_start': Triggered when a node begins execution.
      - 'node_update': Node output deltas and scratchpad logs.
      - 'interrupt': Execution suspended awaiting HITL review.
      - 'final_answer': Synthesized financial report.
      - 'complete': Workflow ended normally.
      - 'error': Runtime failure telemetry.
    """
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")

    # Initial frame
    yield SSEEvent(
        event="lifecycle",
        data={
            "stage": "INITIATED",
            "thread_id": thread_id,
            "trace_id": initial_state.trace_id,
            "input": initial_state.input,
        },
    ).to_sse_packet()

    try:
        # Stream intermediate graph updates
        async for chunk in agent.astream(
            initial_state.model_dump(),
            config=config,
            stream_mode="updates",
        ):
            # chunk is {node_name: state_delta}
            for node_name, delta in chunk.items():
                scratchpad = delta.get("scratchpad", [])
                latest_log = scratchpad[-1] if scratchpad else ""

                yield SSEEvent(
                    event="node_update",
                    data={
                        "node": node_name,
                        "thread_id": thread_id,
                        "latest_log": latest_log,
                        "tool_results_count": len(delta.get("tool_results", [])),
                        "is_terminal": delta.get("is_terminal", False),
                    },
                ).to_sse_packet()

                # If final answer produced
                if "final_answer" in delta and delta["final_answer"]:
                    yield SSEEvent(
                        event="final_answer",
                        data={
                            "thread_id": thread_id,
                            "final_answer": delta["final_answer"],
                            "is_terminal": delta.get("is_terminal", True),
                        },
                    ).to_sse_packet()

        # Check if the execution thread ended in an interrupt state
        curr_state = agent.get_state(config)
        if curr_state and curr_state.next:
            interrupt_data = None
            if hasattr(curr_state, "tasks") and curr_state.tasks:
                for task in curr_state.tasks:
                    if hasattr(task, "interrupts") and task.interrupts:
                        interrupt_data = [i.value for i in task.interrupts]

            yield SSEEvent(
                event="interrupt",
                data={
                    "status": "PENDING_APPROVAL",
                    "thread_id": thread_id,
                    "paused_at_nodes": list(curr_state.next),
                    "interrupt_details": interrupt_data,
                },
            ).to_sse_packet()
        else:
            yield SSEEvent(
                event="complete",
                data={"status": "FINISHED", "thread_id": thread_id},
            ).to_sse_packet()

    except Exception as exc:
        yield SSEEvent(
            event="error",
            data={
                "error_type": type(exc).__name__,
                "message": str(exc),
                "thread_id": thread_id,
            },
        ).to_sse_packet()