from pathlib import Path
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "ui" / "app.py"


def response(payload):
    result = Mock(status_code=200)
    result.json.return_value = payload
    result.raise_for_status.return_value = None
    return result


def test_ready_documents_are_selectable_and_nonready_documents_are_not():
    documents = [
        {
            "document_id": "doc-ready",
            "filename": "annual-2025.pdf",
            "status": "ready",
            "page_count": 96,
        },
        {
            "document_id": "doc-processing",
            "filename": "quarterly.pdf",
            "status": "processing",
            "page_count": None,
        },
    ]

    def request(method, url, **kwargs):
        if url.endswith("/api/v1/config"):
            return response({"auth_mode": "token", "features": {"documents": True}})
        if url.endswith("/api/v1/documents"):
            return response({"items": documents})
        return response({"items": []})

    with patch("requests.request", side_effect=request):
        app = AppTest.from_file(str(APP), default_timeout=15).run()
        app.text_input(key="token_input").set_value("private-token").run()
        selector = app.multiselect(key="document_selection")
        assert selector.options == ["annual-2025.pdf · ready"]
        selector.set_value(["doc-ready"]).run()
        assert app.session_state.selected_document_ids == ["doc-ready"]
        assert any("annual-2025.pdf" in item.value for item in app.markdown)
        assert any("processing" in item.value.lower() for item in app.caption)
        app.text_area(key="query_input").set_value("What changed?")
        with patch(
            "ui.sse_client.stream_sse_query",
            return_value=iter([{"event": "complete", "data": {}}]),
        ) as stream:
            app.button(key="FormSubmitter:query_form-Run analysis").click().run()
        assert stream.call_args.args[1]["document_ids"] == ["doc-ready"]


def test_supabase_login_stores_session_and_refresh_token():
    def request(method, url, **kwargs):
        return response(
            {
                "auth_mode": "supabase",
                "supabase_url": "https://supabase.invalid",
                "supabase_publishable_key": "public-key",
            }
        )

    auth_response = response(
        {
            "access_token": "access",
            "refresh_token": "refresh",
            "user": {"id": "user-1", "email": "analyst@example.com"},
        }
    )
    with (
        patch("requests.request", side_effect=request),
        patch("requests.post", return_value=auth_response),
    ):
        app = AppTest.from_file(str(APP), default_timeout=15).run()
        assert any("Sign in to continue" in item.value for item in app.markdown)
        app.text_input(key="login_email").set_value("analyst@example.com")
        app.text_input(key="login_password").set_value("correct-horse-battery-staple")
        app.button(key="FormSubmitter:login_form-Sign in").click().run()
        assert not app.exception
        assert app.session_state.access_token == "access"
        assert app.session_state.refresh_token == "refresh"


def test_processing_report_can_be_refreshed_into_selectable_sources():
    status = {"value": "processing"}

    def request(method, url, **kwargs):
        if url.endswith("/config"):
            return response({"auth_mode": "token", "features": {"documents": True}})
        if url.endswith("/documents"):
            return response(
                {
                    "items": [
                        {
                            "document_id": "new-report",
                            "filename": "new.pdf",
                            "status": status["value"],
                        }
                    ]
                }
            )
        return response({"items": []})

    with patch("requests.request", side_effect=request):
        app = AppTest.from_file(str(APP), default_timeout=15).run()
        app.text_input(key="token_input").set_value("test-token").run()
        assert app.multiselect(key="document_selection").options == []
        status["value"] = "ready"
        app.button(key="refresh_documents").click().run()
        assert not app.exception
        assert app.multiselect(key="document_selection").options == ["new.pdf · ready"]


def test_saved_research_recovers_cited_answer_after_fresh_session():
    def request(method, url, **kwargs):
        if url.endswith("/config"):
            return response({"auth_mode": "token", "features": {"documents": True}})
        if url.endswith("/threads"):
            return response(
                {"items": [{"thread_id": "saved-1", "input_query": "Revenue?"}]}
            )
        if url.endswith("/threads/saved-1/state"):
            return response(
                {
                    "is_paused": False,
                    "state_values": {
                        "final_answer": "Revenue was USD 42 million.",
                        "answer_status": "grounded",
                        "citations": [
                            {"filename": "annual.pdf", "page": 8, "quote": "Revenue 42"}
                        ],
                    },
                }
            )
        return response({"items": []})

    with patch("requests.request", side_effect=request):
        app = AppTest.from_file(str(APP), default_timeout=15).run()
        app.text_input(key="token_input").set_value("test-token").run()
        assert app.session_state.last_analysis is None
        app.button(key="open_saved_analysis").click().run()
        assert not app.exception
        assert app.session_state.last_analysis["thread_id"] == "saved-1"
        assert app.session_state.last_analysis["citations"][0]["page"] == 8
        assert any("Revenue was USD 42 million." in item.value for item in app.markdown)
        app.text_input(key="token_input").set_value("different-account-token").run()
        assert app.session_state.last_analysis is None
