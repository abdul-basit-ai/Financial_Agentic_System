from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agent.server.schemas import SSEEvent
from agent.state.schema import AgentStateV1

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="legacy-stream")
_SLOTS = threading.BoundedSemaphore(4)


async def stream_agent_events(
    agent: Any, initial_state: AgentStateV1, config: dict[str, Any]
) -> AsyncGenerator[str, None]:
    if not _SLOTS.acquire(blocking=False):
        yield SSEEvent(
            event="error", data={"message": "Streaming capacity is busy."}
        ).to_sse_packet()
        return
    stopped = threading.Event()
    updates: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=16)

    def emit(kind: str, payload: Any) -> None:
        while not stopped.is_set():
            try:
                updates.put((kind, payload), timeout=0.2)
                return
            except queue.Full:
                continue

    def run() -> None:
        try:
            for chunk in agent.stream(
                initial_state.model_dump(), config=config, stream_mode="updates"
            ):
                if stopped.is_set():
                    break
                emit("update", chunk)
            emit("done", None)
        except Exception:
            emit("error", None)
        finally:
            _SLOTS.release()

    future = _EXECUTOR.submit(run)
    started = time.monotonic()
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    try:
        yield SSEEvent(
            event="lifecycle", data={"thread_id": thread_id, "stage": "INITIATED"}
        ).to_sse_packet()
        while time.monotonic() - started < 180:
            try:
                kind, payload = await asyncio.to_thread(updates.get, True, 1)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            if kind == "done":
                snapshot = await asyncio.to_thread(agent.get_state, config)
                event = "interrupt" if snapshot and snapshot.next else "complete"
                yield SSEEvent(
                    event=event, data={"thread_id": thread_id}
                ).to_sse_packet()
                return
            if kind == "error":
                yield SSEEvent(
                    event="error",
                    data={"message": "Execution failed.", "thread_id": thread_id},
                ).to_sse_packet()
                return
            for node, delta in payload.items():
                if not isinstance(delta, dict):
                    continue
                yield SSEEvent(
                    event="node_update", data={"node": node, "thread_id": thread_id}
                ).to_sse_packet()
                if delta.get("final_answer"):
                    yield SSEEvent(
                        event="final_answer",
                        data={
                            k: delta.get(k)
                            for k in (
                                "final_answer",
                                "citations",
                                "answer_status",
                                "calculation",
                            )
                        },
                    ).to_sse_packet()
        yield SSEEvent(
            event="error", data={"message": "Streaming deadline exceeded."}
        ).to_sse_packet()
    finally:
        stopped.set()
        if future.cancel():
            _SLOTS.release()
