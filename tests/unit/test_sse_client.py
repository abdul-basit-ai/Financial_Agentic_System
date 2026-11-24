"""Tests for the Streamlit-side SSE parser."""

from collections.abc import Iterator
from typing import Any

from ui.sse_client import stream_sse_query


class _FakeResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self) -> Iterator[bytes]:
        yield from self._lines


def test_stream_parser_flushes_final_event_without_blank_line(monkeypatch: Any) -> None:
    response = _FakeResponse([b"event: complete", b'data: {"status": "FINISHED"}'])
    captured: dict[str, Any] = {}

    def post(*args: Any, **kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return response

    monkeypatch.setattr("ui.sse_client.requests.post", post)

    events = list(
        stream_sse_query(
            "http://api:8000",
            {"query": "Revenue?"},
            headers={"Authorization": "Bearer user-token"},
        )
    )

    assert events == [{"event": "complete", "data": {"status": "FINISHED"}}]
    assert captured["headers"]["Authorization"] == "Bearer user-token"
    assert captured["headers"]["Accept"] == "text/event-stream"
    assert captured["timeout"] == (5.0, 180.0)
