"""Server-Sent Events (SSE) streaming engine for LangGraph execution.

The compiled graph runs with a SYNC checkpointer (RedisSaver implements no
async checkpoint methods — BaseCheckpointSaver.aput raises
NotImplementedError), so ``agent.astream`` breaks the moment it tries to
persist a checkpoint. The graph therefore runs ``agent.stream`` on a worker
thread and forwards update chunks to this async generator over an asyncio
queue: SSE semantics preserved, any sync-only checkpointer supported.

Emits:
  - 'lifecycle': Start/initialization metadata.
  - 'node_update': Node output deltas and scratchpad logs.
  - 'interrupt': Execution suspended awaiting HITL review.
  - 'final_answer': Synthesized financial report.
  - 'complete': Workflow ended normally.
  - 'error': Runtime failure telemetry.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator
from typing import Any

from agent.server.schemas import SSEEvent
from agent.state.schema import AgentStateV1


async def stream_agent_events(
    agent: Any,
    initial_state: AgentStateV1,
    config: dict[str, Any],
) -> AsyncGenerator[str, None]:
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

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    def _put(kind: str, payload: Any) -> None:
        try:
            asyncio.run_coroutine_threadsafe(queue.put((kind, payload)), loop)
        except RuntimeError:
            # Event loop closed (app shutdown mid-stream); nothing to forward.
            pass

    def _run_graph() -> None:
        try:
            for chunk in agent.stream(
                initial_state.model_dump(), config=config, stream_mode="updates"
            ):
                _put("chunk", chunk)
        except Exception as exc:  # noqa: BLE001 — surfaced verbatim to the client
            _put("error", exc)
        finally:
            _put("done", None)

    worker = threading.Thread(target=_run_graph, name="finagent-stream", daemon=True)
    worker.start()

    stream_error: Exception | None = None
    while True:
        kind, payload = await queue.get()
        if kind == "done":
            break
        if kind == "error":
            stream_error = payload
            break

        # chunk is {node_name: state_delta}
        for node_name, delta in payload.items():
            if not isinstance(delta, dict):
                continue
            scratchpad = delta.get("scratchpad", [])
            latest_log = scratchpad[-1] if scratchpad else ""

            yield SSEEvent(
                event="node_update",
                data={
                    "node": node_name,
                    "thread_id": thread_id,
                    "latest_log": latest_log,
                    "tool_results_count": len(delta.get("tool_results", []) or []),
                    "is_terminal": delta.get("is_terminal", False),
                },
            ).to_sse_packet()

            if delta.get("final_answer"):
                yield SSEEvent(
                    event="final_answer",
                    data={
                        "thread_id": thread_id,
                        "final_answer": delta["final_answer"],
                        "is_terminal": delta.get("is_terminal", True),
                    },
                ).to_sse_packet()

    if stream_error is not None:
        yield SSEEvent(
            event="error",
            data={
                "error_type": type(stream_error).__name__,
                "message": str(stream_error),
                "thread_id": thread_id,
            },
        ).to_sse_packet()
        return

    # Interrupt vs complete: get_state hits Redis synchronously.
    curr_state = await asyncio.to_thread(agent.get_state, config)
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
