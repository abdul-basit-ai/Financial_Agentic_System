from pathlib import Path
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "ui" / "app.py"


def response(data=None, status_code=200):
    result = Mock(status_code=status_code)
    result.json.return_value = data or {}
    result.raise_for_status.return_value = None
    return result


def api_request(method, url, **kwargs):
    if url.endswith("/api/v1/config"):
        return response({"auth_mode": "token", "features": {"documents": True}})
    if url.endswith("/api/v1/documents"):
        return response({"items": []})
    if url.endswith("/api/v1/approvals"):
        return response({"items": []})
    return response()


def test_analysis_persists_answer_citations_and_selected_report_ids():
    events = [
        {"event": "lifecycle", "data": {}},
        {
            "event": "final_answer",
            "data": {
                "final_answer": "Debt increased by 12%.",
                "citations": [
                    {
                        "filename": "annual.pdf",
                        "page": 42,
                        "quote": "Long-term debt maturities",
                    }
                ],
                "answer_status": "supported",
            },
        },
        {"event": "complete", "data": {}},
    ]
    with (
        patch("requests.request", side_effect=api_request),
        patch("ui.sse_client.stream_sse_query", return_value=iter(events)) as stream,
    ):
        app = AppTest.from_file(str(APP), default_timeout=15).run()
        app.text_input(key="token_input").set_value("test-user-token").run()
        app.text_input(key="company_input").set_value("ETR")
        app.text_area(key="query_input").set_value("What changed?")
        app.button(key="FormSubmitter:query_form-Run analysis").click().run()
        assert not app.exception
        assert stream.call_args.args[1]["company_identifier"] == "ETR"
        assert stream.call_args.args[1]["document_ids"] == []
        assert stream.call_args.kwargs["headers"] == {
            "Authorization": "Bearer test-user-token"
        }
        record = app.session_state.last_analysis
        assert record["final_answer"] == "Debt increased by 12%."
        assert record["citations"][0]["page"] == 42
        app.run()
        assert any("Debt increased by 12%." in item.value for item in app.markdown)
        assert any("annual.pdf" in item.label for item in app.expander)


def test_review_resume_updates_matching_analysis_from_response():
    case = {
        "thread_id": "review-test",
        "company_identifier": "ETR",
        "input_query": "Check debt",
        "paused_nodes": ["compute"],
        "trigger_reasons": ["Low confidence"],
        "scratchpad_summary": [],
        "pending_tool_calls": [],
    }
    resume = response(
        {
            "final_answer": "Verified from the filing.",
            "citations": [{"filename": "annual.pdf", "page": 7}],
            "answer_status": "supported",
            "is_terminal": True,
        }
    )

    def api(method, url, **kwargs):
        if url.endswith("/resume"):
            return resume
        if url.endswith("/approvals"):
            return response({"items": [case]})
        return response({"items": []})

    with patch("requests.request", side_effect=api) as request:
        app = AppTest.from_file(str(APP), default_timeout=15).run()
        app.text_input(key="token_input").set_value("test-user-token").run()
        app.radio(key="action_review-test").set_value("APPROVE").run()
        app.button(key="submit_review-test").click().run()
        assert not app.exception
        resume_call = next(
            call for call in request.call_args_list if call.args[1].endswith("/resume")
        )
        assert resume_call.kwargs["json"]["action"] == "APPROVE"
        record = app.session_state.last_analysis
        assert record["final_answer"] == "Verified from the filing."
        assert record["paused"] is False
        assert record["status"] == "complete"


def test_invalid_override_is_disabled():
    case = {
        "thread_id": "override-test",
        "company_identifier": "ETR",
        "input_query": "Check debt",
        "trigger_reasons": [],
        "scratchpad_summary": [],
        "pending_tool_calls": [],
    }
    with patch(
        "requests.request",
        side_effect=lambda method, url, **kwargs: (
            response({"items": [case]})
            if url.endswith("/approvals")
            else response({"items": []})
        ),
    ):
        app = AppTest.from_file(str(APP), default_timeout=15).run()
        app.text_input(key="token_input").set_value("test-user-token").run()
        app.radio(key="action_override-test").set_value("OVERRIDE").run()
        app.text_area(key="override_override-test").set_value("invalid json").run()
        assert app.button(key="submit_override-test").disabled


def test_offline_workspace_has_connection_state():
    with patch("requests.request", side_effect=ConnectionError("Unavailable")):
        app = AppTest.from_file(str(APP), default_timeout=15).run()
        assert not app.exception
        assert any("API is not available" in item.value for item in app.warning)


def test_server_token_is_not_forwarded_to_development_override(monkeypatch):
    monkeypatch.setenv("FINAGENT_ALLOW_API_OVERRIDE", "1")
    monkeypatch.setenv("API_URL", "http://trusted.internal:8000")
    monkeypatch.setenv("FINAGENT_API_TOKEN", "server-only-token")
    sent = []

    def request(method, url, **kwargs):
        sent.append((url, kwargs.get("headers", {})))
        if url.endswith("/api/v1/config"):
            return response({"auth_mode": "token", "features": {"documents": True}})
        return response({"items": []})

    with patch("requests.request", side_effect=request):
        app = AppTest.from_file(str(APP), default_timeout=15).run()
        app.text_input(key="api_base_url").set_value("http://untrusted.invalid").run()

    forwarded = [
        headers for url, headers in sent if url.startswith("http://untrusted.invalid")
    ]
    assert forwarded
    assert all("Authorization" not in headers for headers in forwarded)


def test_insufficient_answer_status_is_presented_as_abstention():
    events = [
        {
            "event": "final_answer",
            "data": {
                "final_answer": "The filing does not support a reliable figure.",
                "citations": [],
                "answer_status": "insufficient",
            },
        }
    ]
    with (
        patch("requests.request", side_effect=api_request),
        patch("ui.sse_client.stream_sse_query", return_value=iter(events)),
    ):
        app = AppTest.from_file(str(APP), default_timeout=15).run()
        app.text_input(key="token_input").set_value("test-user-token").run()
        app.button(key="FormSubmitter:query_form-Run analysis").click().run()

    assert any(
        "insufficient" in item.value.lower() or "insufficient" in item.label.lower()
        for item in app.warning
    )
