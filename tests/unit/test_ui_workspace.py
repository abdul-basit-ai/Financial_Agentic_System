"""Exercise the research and review flows without a running gateway."""

from pathlib import Path
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "ui" / "app.py"


def response(items=None):
    result = Mock(status_code=200)
    result.json.return_value = {"items": items or []}
    return result


def test_analysis_and_persistence():
    events = [
        {"event": "lifecycle", "data": {}},
        {"event": "final_answer", "data": {"final_answer": "Debt increased by 12%."}},
        {"event": "complete", "data": {}},
    ]
    with (
        patch("requests.get", return_value=response()),
        patch("ui.sse_client.stream_sse_query", return_value=iter(events)) as stream,
    ):
        app = AppTest.from_file(str(APP)).run()
        assert not app.exception
        assert not app.tabs
        app.button(key="preset_etr_rider").click().run()
        assert "rider revenue" in app.text_area(key="query_input").value
        app.button(key="FormSubmitter:query_form-Run analysis").click().run()
        assert not app.exception
        assert stream.call_args.args[1]["company_identifier"] == "ETR"
        assert (
            app.session_state.last_analysis["final_answer"] == "Debt increased by 12%."
        )
        app.run()
        assert any("Debt increased by 12%." in m.value for m in app.markdown)


def test_review_and_invalid_override():
    case = {
        "thread_id": "review-test",
        "company_identifier": "ETR",
        "input_query": "Check debt",
        "paused_nodes": ["compute"],
        "trigger_reasons": ["Low confidence"],
        "scratchpad_summary": [],
        "pending_tool_calls": [],
    }
    with (
        patch("requests.get", return_value=response([case])),
        patch("requests.post", return_value=response()) as post,
    ):
        app = AppTest.from_file(str(APP)).run()
        assert not app.exception
        app.radio(key="action_review-test").set_value("OVERRIDE").run()
        app.text_area(key="override_review-test").set_value("invalid json").run()
        assert app.button(key="btn_review-test").disabled
        app.radio(key="action_review-test").set_value("APPROVE").run()
        app.button(key="btn_review-test").click().run()
        assert not app.exception
        assert post.call_args.kwargs["json"]["action"] == "APPROVE"


def test_offline_workspace():
    with patch("requests.get", side_effect=ConnectionError("Unavailable")):
        app = AppTest.from_file(str(APP)).run()
        assert not app.exception
        assert any("review desk is offline" in info.value for info in app.info)
