"""Lightweight Server-Sent Events (SSE) streaming client for the Streamlit UI."""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any
import requests


def stream_sse_query(
    api_url: str,
    payload: dict[str, Any],
    timeout_seconds: float = 60.0,
) -> Generator[dict[str, Any], None, None]:
    """Streams Server-Sent Events from the FastAPI backend.

    Yields dictionaries with format: {"event": str, "data": dict[str, Any]}.
    """
    endpoint = f"{api_url.rstrip('/')}/api/v1/query/stream"

    with requests.post(
        endpoint,
        json=payload,
        stream=True,
        headers={"Accept": "text/event-stream"},
        timeout=timeout_seconds,
    ) as response:
        response.raise_for_status()

        current_event = "message"
        current_data_lines: list[str] = []

        for raw_line in response.iter_lines():
            if not raw_line:
                # Dispatch event upon double newline
                if current_data_lines:
                    raw_data = "\n".join(current_data_lines)
                    try:
                        parsed_json = json.loads(raw_data)
                    except Exception:
                        parsed_json = {"raw": raw_data}

                    yield {"event": current_event, "data": parsed_json}
                    current_event = "message"
                    current_data_lines = []
                continue

            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line

            if line.startswith("event:"):
                current_event = line.replace("event:", "", 1).strip()
            elif line.startswith("data:"):
                current_data_lines.append(line.replace("data:", "", 1).strip())