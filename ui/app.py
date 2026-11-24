from __future__ import annotations

import html
import json
import math
import os
import uuid
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any

import requests
import streamlit as st

try:
    stream_sse_query = import_module("ui.sse_client").stream_sse_query
except ModuleNotFoundError:
    stream_sse_query = import_module("sse_client").stream_sse_query


st.set_page_config(
    page_title="Ledger | Financial research",
    page_icon=":material/query_stats:",
    layout="wide",
    initial_sidebar_state="auto",
)

CSS = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

DEFAULT_API_URL = os.getenv("API_URL") or "http://127.0.0.1:8000"
ALLOW_API_OVERRIDE = os.getenv("FINAGENT_ALLOW_API_OVERRIDE", "0").strip() == "1"


def _api_url() -> str:
    selected = st.session_state.get("api_base_url", DEFAULT_API_URL)
    return str(selected if ALLOW_API_OVERRIDE else DEFAULT_API_URL).rstrip("/")


def _headers() -> dict[str, str]:
    token = str(st.session_state.get("access_token") or "").strip()
    server_token = os.getenv("FINAGENT_API_TOKEN", "").strip()
    if (
        server_token
        and token == server_token
        and _api_url() != DEFAULT_API_URL.rstrip("/")
    ):
        return {}
    return {"Authorization": f"Bearer {token}"} if token else {}


def _request(method: str, path: str, **kwargs: Any) -> requests.Response:
    headers = dict(kwargs.pop("headers", {}) or {})
    headers.update(_headers())
    response = requests.request(
        method,
        f"{_api_url()}{path}",
        headers=headers,
        timeout=kwargs.pop("timeout", 10.0),
        **kwargs,
    )
    if response.status_code == 401 and st.session_state.get("refresh_token"):
        refreshed = requests.post(
            f"{_api_url()}/api/v1/auth/refresh",
            json={"refresh_token": st.session_state.refresh_token},
            timeout=10.0,
        )
        if refreshed.ok:
            auth = refreshed.json()
            st.session_state.access_token = auth.get("access_token", "")
            if auth.get("refresh_token"):
                st.session_state.refresh_token = auth["refresh_token"]
            headers.update(_headers())
            response = requests.request(
                method,
                f"{_api_url()}{path}",
                headers=headers,
                timeout=kwargs.pop("timeout", 10.0),
                **kwargs,
            )
    return response


def _get_json(path: str, timeout: float = 10.0) -> dict[str, Any]:
    response = _request("GET", path, timeout=timeout)
    response.raise_for_status()
    result = response.json()
    return result if isinstance(result, dict) else {}


def _format_date(value: Any) -> str:
    if not value:
        return ""
    return str(value).replace("T", " ").replace("Z", " UTC")[:19]


def _citation_label(citation: dict[str, Any], index: int) -> str:
    filename = str(citation.get("filename") or "Source document")
    page = citation.get("page")
    location = f"p. {page}" if page is not None else "Page not specified"
    table = citation.get("table_id")
    if table:
        location += f" · Table {table}"
    return f"[{index}] {filename} · {location}"


def _render_citations(citations: Any) -> None:
    if not isinstance(citations, list) or not citations:
        st.caption("No page-level evidence was attached to this answer.")
        return
    for index, raw in enumerate(citations, start=1):
        if not isinstance(raw, dict):
            continue
        label = _citation_label(raw, index)
        with st.expander(label):
            row = raw.get("row_label")
            column = raw.get("column_label")
            if row or column:
                st.caption(" · ".join(str(part) for part in (row, column) if part))
            quote = raw.get("quote")
            if quote:
                st.markdown(f"> {str(quote)}")
            evidence_id = raw.get("evidence_id")
            if evidence_id:
                st.caption(f"Evidence reference: {evidence_id}")


def _render_analysis(record: dict[str, Any]) -> None:
    answer = record.get("final_answer")
    if answer:
        st.markdown("### Answer")
        st.markdown(str(answer))
        status = str(record.get("answer_status") or "").lower()
        if status in {"abstain", "abstained", "insufficient", "insufficient_evidence"}:
            st.warning(
                "The available filing evidence was insufficient for a reliable answer."
            )
        calculation = record.get("calculation")
        if calculation:
            with st.expander("Calculation details"):
                st.code(str(calculation), language="text")
        st.markdown("### Filing evidence")
        _render_citations(record.get("citations"))
    elif record.get("paused"):
        st.info(
            "This analysis is paused for review. Submit a decision in the review desk to continue."
        )
    elif record.get("error"):
        st.error(f"The latest analysis failed: {record['error']}")
    elif record.get("status") == "running":
        st.info(
            "The analysis is still running. Refresh this page to check for a result."
        )
    elif record.get("answer_status"):
        st.info(
            f"No final answer was returned. Result status: {record['answer_status']}."
        )
    elif record.get("status") == "complete":
        st.info("The review decision is complete. No answer was returned.")


def _load_config() -> dict[str, Any]:
    try:
        return _get_json("/api/v1/config", timeout=3.0)
    except Exception as exc:
        return {"_error": str(exc), "auth_mode": "token", "features": {}}


if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = DEFAULT_API_URL
if "analyses" not in st.session_state:
    st.session_state.analyses = []
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None
if "selected_document_ids" not in st.session_state:
    st.session_state.selected_document_ids = []
if "workspace_config" not in st.session_state:
    st.session_state.workspace_config = None


with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">&#9672; &nbsp; ledger<span>FINANCIAL RESEARCH</span></div>',
        unsafe_allow_html=True,
    )
    library_panel = st.container()
    history_panel = st.container()
    with st.expander("Account & connection"):
        account_controls = st.container()
        if ALLOW_API_OVERRIDE:
            st.text_input("Backend API URL", key="api_base_url")
        else:
            st.caption(f"Backend API: {DEFAULT_API_URL}")
        if st.button("Reload connection", use_container_width=True):
            st.session_state.workspace_config = None
            st.rerun()
        if st.button(
            "Sign out",
            use_container_width=True,
            disabled=not bool(st.session_state.get("access_token")),
        ):
            for key in (
                "access_token",
                "refresh_token",
                "user",
                "selected_document_ids",
                "token_input",
                "analyses",
                "last_analysis",
                "workspace_identity",
            ):
                st.session_state.pop(key, None)
            st.rerun()

config = st.session_state.workspace_config
if config is None:
    config = _load_config()
    st.session_state.workspace_config = config
auth_mode = str(config.get("auth_mode") or "token").lower()
backend_error = config.get("_error")

if auth_mode == "supabase" and not st.session_state.get("access_token"):
    st.markdown(
        '<div class="topbar"><span>LEDGER / PRIVATE WORKSPACE</span><span>Sign in to continue</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero"><div class="eyebrow">FINANCIAL RESEARCH</div><h1>Your filings, your evidence.</h1><p>Sign in to access your private reports and saved research.</p></div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        with st.form("login_form"):
            email = st.text_input("Email address", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            login = st.form_submit_button(
                "Sign in", type="primary", use_container_width=True
            )
        if login:
            try:
                response = requests.post(
                    f"{_api_url()}/api/v1/auth/login",
                    json={"email": email, "password": password},
                    timeout=15.0,
                )
                response.raise_for_status()
                auth = response.json()
                st.session_state.access_token = auth.get("access_token", "")
                st.session_state.refresh_token = auth.get("refresh_token")
                st.session_state.user = auth.get("user", {})
                st.rerun()
            except Exception as exc:
                st.error(f"Sign in failed: {exc}")
    st.stop()

if auth_mode != "supabase":
    env_token = os.getenv("FINAGENT_API_TOKEN", "")
    with account_controls:
        st.text_input("Workspace access token", type="password", key="token_input")
    st.session_state.access_token = st.session_state.get("token_input") or env_token

workspace_identity = sha256(
    str(st.session_state.get("access_token") or "").encode()
).hexdigest()
if st.session_state.get("workspace_identity") != workspace_identity:
    st.session_state.analyses = []
    st.session_state.last_analysis = None
    st.session_state.selected_document_ids = []
    st.session_state.pop("document_selection", None)
    st.session_state.workspace_identity = workspace_identity

st.markdown(
    '<div class="topbar"><span>YOUR RESEARCH WORKSPACE</span>'
    f'<span class="connection">{html.escape(str((st.session_state.get("user") or {}).get("email") or "Private workspace"))}</span></div>'
    '<div class="hero"><div class="eyebrow">THE RESEARCH DESK</div>'
    "<h1>Clarity behind the numbers.</h1>"
    "<p>Ask about your reports. Check every answer against the source.</p></div>",
    unsafe_allow_html=True,
)

if backend_error:
    st.warning(
        "The workspace API is not available. Check the connection settings and try again."
    )
    with st.expander("Connection details"):
        st.caption(str(backend_error))

documents: list[dict[str, Any]] = []
documents_error = None
if not backend_error and st.session_state.get("access_token"):
    try:
        document_response = _get_json("/api/v1/documents")
        raw_documents = document_response.get("items", [])
        documents = (
            [item for item in raw_documents if isinstance(item, dict)]
            if isinstance(raw_documents, list)
            else []
        )
    except Exception as exc:
        documents_error = str(exc)

with library_panel:
    st.markdown("### Your reports")
    if st.button("Refresh reports", key="refresh_documents", use_container_width=False):
        st.rerun()

    if documents_error:
        st.info(
            "Your filing library could not be loaded. Check the connection or sign in again."
        )
        with st.expander("Connection details"):
            st.caption(documents_error)
    elif documents:
        pending_documents = {
            str(doc.get("document_id")): str(doc.get("status"))
            for doc in documents
            if doc.get("status") in {"queued", "processing"}
        }
        if pending_documents:

            @st.fragment(run_every=5)
            def refresh_processing_reports() -> None:
                try:
                    latest = _get_json("/api/v1/documents")
                    statuses = {
                        str(doc.get("document_id")): str(doc.get("status"))
                        for doc in latest.get("items", [])
                        if isinstance(doc, dict)
                    }
                    if any(
                        statuses.get(key) != value
                        for key, value in pending_documents.items()
                    ):
                        st.rerun()
                except requests.RequestException:
                    pass
                st.caption("Processing reports. Status updates automatically.")

            refresh_processing_reports()
        selected_ids = set(st.session_state.selected_document_ids)
        ready_docs = [
            doc
            for doc in documents
            if str(doc.get("status", "")).lower()
            in {"ready", "completed", "complete", "indexed"}
        ]
        labels = {
            str(
                doc.get("document_id")
            ): f"{doc.get('filename', 'Untitled report')} · {doc.get('status', 'unknown')}"
            for doc in documents
        }
        selected = st.multiselect(
            "Reports to use for research",
            options=[str(doc.get("document_id")) for doc in ready_docs],
            default=[
                doc_id
                for doc_id in selected_ids
                if doc_id in labels
                and doc_id in {str(doc.get("document_id")) for doc in ready_docs}
            ],
            format_func=lambda document_id: labels.get(document_id, document_id),
            key="document_selection",
            help="Only ready reports can be selected for analysis.",
        )
        st.session_state.selected_document_ids = selected
        with st.expander(f"All reports · {len(documents)}"):
            for doc in documents:
                status = str(doc.get("status") or "unknown")
                st.markdown(f"**{doc.get('filename', 'Untitled report')}**")
                pages = doc.get("page_count")
                st.caption(
                    f"{status.capitalize()} | {pages} pages"
                    if pages
                    else status.capitalize()
                )
                if doc.get("error"):
                    st.caption(str(doc["error"]))
    elif not documents_error and st.session_state.get("access_token"):
        st.markdown(
            '<div class="empty-state"><h3>No reports yet</h3><p>Upload a PDF filing to create a searchable, page-referenced source for your research.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "Sign in or enter a workspace access token to load your private reports."
        )

    with st.expander("Upload a report", expanded=not bool(documents)):
        st.caption(
            "Upload annual reports and SEC filings in PDF format. Documents are private to your account."
        )
        max_upload_bytes = int(config.get("max_upload_bytes") or 25 * 1024 * 1024)
        uploaded = st.file_uploader(
            "Add a filing",
            type=["pdf"],
            accept_multiple_files=False,
            key="report_upload",
            max_upload_size=max(1, math.ceil(max_upload_bytes / (1024 * 1024))),
        )
        company_for_upload = st.text_input(
            "Company identifier (optional)",
            key="upload_company",
            placeholder="Ticker or company name",
        )
        if uploaded is not None:
            st.caption(
                f"Selected file: {uploaded.name} · {uploaded.size / (1024 * 1024):.1f} MiB · Maximum {max_upload_bytes / (1024 * 1024):.1f} MiB"
            )
            too_large = uploaded.size > max_upload_bytes
            if too_large:
                st.warning("This PDF exceeds the workspace upload limit.")
            if st.button(
                "Upload report",
                type="primary",
                key="upload_document",
                disabled=too_large or not bool(st.session_state.get("access_token")),
            ):
                try:
                    upload_response = _request(
                        "POST",
                        "/api/v1/documents",
                        files={
                            "file": (
                                uploaded.name,
                                uploaded.getvalue(),
                                "application/pdf",
                            )
                        },
                        data=(
                            {"company_identifier": company_for_upload.strip()}
                            if company_for_upload.strip()
                            else {}
                        ),
                        timeout=60.0,
                    )
                    upload_response.raise_for_status()
                    item = upload_response.json()
                    st.success(
                        f"{item.get('filename', uploaded.name)} added to your filing library."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Upload failed: {exc}")

with history_panel:
    with st.expander("Saved research"):
        if st.session_state.get("access_token"):
            try:
                saved = _get_json("/api/v1/threads").get("items", [])
                saved = [item for item in saved if isinstance(item, dict)]
                by_id = {
                    str(item["thread_id"]): item
                    for item in saved
                    if item.get("thread_id")
                }
                if by_id:
                    selected_thread = st.selectbox(
                        "Saved analysis",
                        options=list(by_id),
                        format_func=lambda key: str(
                            by_id[key].get("input_query") or key
                        )[:100],
                        key="saved_thread",
                    )
                    if st.button("Open saved analysis", key="open_saved_analysis"):
                        snapshot = _get_json(f"/api/v1/threads/{selected_thread}/state")
                        values = snapshot.get("state_values") or {}
                        summary = by_id[selected_thread]
                        recovered = {
                            **values,
                            "thread_id": selected_thread,
                            "query": summary.get("input_query"),
                            "company": summary.get("company_identifier"),
                            "paused": bool(snapshot.get("is_paused")),
                            "status": (
                                "paused"
                                if snapshot.get("is_paused")
                                else (
                                    "complete"
                                    if values.get("final_answer")
                                    else "running"
                                )
                            ),
                        }
                        st.session_state.last_analysis = recovered
                        st.session_state.analyses = [
                            recovered,
                            *[
                                item
                                for item in st.session_state.analyses
                                if item.get("thread_id") != selected_thread
                            ],
                        ]
                        st.rerun()
                else:
                    st.caption("Completed and running analyses will appear here.")
            except Exception as exc:
                st.caption(f"Saved research is unavailable: {exc}")
        else:
            st.caption("Sign in to load saved research.")


research, review = st.tabs(["Research", "Review"])

with research:
    selected_reports = [
        str(doc.get("filename") or "Untitled report")
        for doc in documents
        if doc.get("document_id") in st.session_state.selected_document_ids
    ]
    source_label = (
        ", ".join(html.escape(name) for name in selected_reports)
        if selected_reports
        else "Choose reports in the sidebar to get started."
    )
    st.markdown(
        f'<div class="source-summary"><strong>{len(selected_reports)} {"report" if len(selected_reports) == 1 else "reports"} selected</strong><span>{source_label}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Ask about your reports")
    with st.form("query_form", clear_on_submit=False):
        user_query = st.text_area(
            "Your question",
            placeholder="For example: How did operating income change from 2020 to 2021?",
            height=140,
            key="query_input",
        )
        with st.expander("Company filter (optional)"):
            company_id = st.text_input(
                "Company identifier",
                key="company_input",
                placeholder="Ticker or company name",
            )
        submitted = st.form_submit_button(
            "Run analysis",
            use_container_width=True,
            type="primary",
            disabled=not bool(st.session_state.get("access_token")),
        )
    if submitted and not user_query.strip():
        st.warning("Enter a financial question before running the analysis.")
    if submitted and user_query.strip():
        thread_id = f"ui_{uuid.uuid4().hex}"
        record: dict[str, Any] = {
            "thread_id": thread_id,
            "company": company_id,
            "query": user_query,
            "document_ids": list(st.session_state.selected_document_ids),
            "status": "running",
            "final_answer": None,
            "citations": [],
            "paused": False,
            "error": None,
            "timeline": [],
        }
        with st.container(border=True):
            st.markdown("#### Research activity")
            with st.status(
                "Searching selected filing evidence…", expanded=True
            ) as status_box:
                try:
                    payload = {
                        "query": user_query,
                        "company_identifier": company_id,
                        "thread_id": thread_id,
                        "document_ids": list(st.session_state.selected_document_ids),
                    }
                    for event in stream_sse_query(
                        _api_url(), payload, headers=_headers()
                    ):
                        event_type = event.get("event")
                        data = event.get("data", {})
                        if not isinstance(data, dict):
                            data = {}
                        if event_type == "lifecycle":
                            status_box.update(label="Searching filing evidence…")
                            record["timeline"].append("Evidence search started")
                        elif event_type == "node_update":
                            detail = str(
                                data.get("latest_log")
                                or data.get("node")
                                or "Research step completed"
                            )
                            record["timeline"].append(detail)
                            status_box.update(label=detail[:120])
                        elif event_type == "interrupt":
                            record["paused"] = True
                            record["status"] = "paused"
                            status_box.update(label="Paused for review", state="error")
                        elif event_type == "final_answer":
                            record["final_answer"] = data.get("final_answer")
                            record["citations"] = data.get("citations") or []
                            record["answer_status"] = data.get("answer_status")
                            record["calculation"] = data.get("calculation")
                            record["status"] = (
                                "paused" if record["paused"] else "complete"
                            )
                        elif event_type == "complete":
                            if not record["paused"]:
                                record["status"] = "complete"
                                status_box.update(
                                    label="Analysis complete", state="complete"
                                )
                        elif event_type == "error":
                            record["error"] = str(
                                data.get("message") or "Execution failed."
                            )
                            record["status"] = "error"
                            status_box.update(label="Analysis failed", state="error")
                except Exception as exc:
                    record["error"] = str(exc)
                    record["status"] = "error"
                    status_box.update(
                        label="Could not complete analysis", state="error"
                    )
        st.session_state.analyses.insert(0, record)
        st.session_state.last_analysis = record
        st.rerun()

    last_analysis = st.session_state.last_analysis
    if isinstance(last_analysis, dict):
        st.markdown(
            f"<div class='session-meta'><strong>Latest analysis</strong><span>{html.escape(str(last_analysis.get('query') or 'Saved research'))}</span></div>",
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            _render_analysis(last_analysis)
        timeline = last_analysis.get("timeline")
        if timeline:
            with st.expander("Research activity"):
                for item in timeline:
                    st.markdown(f"- {item}")
    else:
        st.markdown(
            '<div class="empty-state"><div class="empty-mark">↗</div><h3>Start with a filing-backed question</h3><p>Select one or more processed reports, then ask about a figure, trend, or disclosure. Results include page references and supporting excerpts.</p></div>',
            unsafe_allow_html=True,
        )

    analyses = st.session_state.analyses
    if isinstance(analyses, list) and len(analyses) > 1:
        with st.expander(f"Recent research · {len(analyses)} sessions"):
            for old_record in analyses[1:]:
                if not isinstance(old_record, dict):
                    continue
                label = f"{old_record.get('query', 'Research session')[:90]} · {old_record.get('status', 'unknown')}"
                with st.expander(label):
                    st.caption(
                        f"{old_record.get('company') or 'Company not specified'} · {old_record.get('thread_id', '')}"
                    )
                    _render_analysis(old_record)

with review:
    st.markdown("### Review desk")
    st.caption("Review paused decisions and resume the original analysis.")
    pending: list[dict[str, Any]] = []
    review_error = None
    if st.session_state.get("access_token"):
        try:
            approval_data = _get_json("/api/v1/approvals")
            raw_items = approval_data.get("items", [])
            pending = (
                [item for item in raw_items if isinstance(item, dict)]
                if isinstance(raw_items, list)
                else []
            )
        except Exception as exc:
            review_error = str(exc)
    else:
        review_error = "Sign in to load review decisions."
    st.markdown(
        f'<div class="kpi"><div class="kpi-number">{len(pending)}</div><div class="kpi-label">Waiting for review</div></div>',
        unsafe_allow_html=True,
    )
    if st.button("Refresh review queue", use_container_width=True):
        st.rerun()
    if review_error:
        st.info("The review queue is unavailable. Check your sign-in and connection.")
        with st.expander("Connection details"):
            st.caption(review_error)
    elif not pending:
        st.success("No decisions are waiting for review.")
    for index, item in enumerate(pending, start=1):
        thread_id = str(item.get("thread_id", "unknown"))
        with st.expander(
            f"Review {index} · {item.get('company_identifier') or 'Company'} · {thread_id}",
            expanded=True,
        ):
            st.markdown("**Question**")
            st.write(item.get("input_query", ""))
            st.markdown("**Flagged for**")
            reasons = item.get("trigger_reasons") or ["Manual review required"]
            st.markdown(" · ".join(str(reason) for reason in reasons))
            with st.expander("Recent reasoning context"):
                for entry in item.get("scratchpad_summary") or []:
                    st.write(entry)
            action = st.radio(
                "Decision",
                ["APPROVE", "OVERRIDE", "REJECT"],
                key=f"action_{thread_id}",
                horizontal=True,
            )
            analyst = st.text_input(
                "Reviewer", value="analyst", key=f"analyst_{thread_id}"
            )
            feedback = st.text_area(
                "Rationale",
                key=f"feedback_{thread_id}",
                placeholder="Record the reason for this decision.",
            )
            overrides: dict[str, Any] = {}
            valid_override = True
            if action == "OVERRIDE":
                default = {"tool_calls": item.get("pending_tool_calls") or []}
                override_raw = st.text_area(
                    "Edited calls (JSON)",
                    value=json.dumps(default, indent=2, default=str),
                    key=f"override_{thread_id}",
                    height=220,
                )
                try:
                    overrides = json.loads(override_raw)
                    calls = overrides.get("tool_calls")
                    if not isinstance(calls, list) or not all(
                        isinstance(call, dict)
                        and "task_id" in call
                        and "target_tool" in call
                        for call in calls
                    ):
                        valid_override = False
                        st.error(
                            "Each edited call needs task_id and target_tool fields."
                        )
                except (ValueError, TypeError):
                    valid_override = False
                    st.error("Enter a valid JSON object containing a tool_calls list.")
            if st.button(
                "Submit decision",
                key=f"submit_{thread_id}",
                type="primary",
                use_container_width=True,
                disabled=not valid_override,
            ):
                try:
                    resume_response = _request(
                        "POST",
                        f"/api/v1/threads/{thread_id}/resume",
                        json={
                            "action": action,
                            "analyst_id": analyst,
                            "feedback": feedback,
                            "overrides": overrides,
                        },
                        timeout=45.0,
                    )
                    resume_response.raise_for_status()
                    result = resume_response.json()
                    matching = next(
                        (
                            entry
                            for entry in st.session_state.analyses
                            if isinstance(entry, dict)
                            and entry.get("thread_id") == thread_id
                        ),
                        None,
                    )
                    if matching is None:
                        matching = {
                            "thread_id": thread_id,
                            "company": item.get("company_identifier"),
                            "query": item.get("input_query", ""),
                            "timeline": [],
                        }
                        st.session_state.analyses.insert(0, matching)
                    matching["final_answer"] = result.get("final_answer")
                    matching["citations"] = result.get("citations") or []
                    matching["answer_status"] = result.get("answer_status")
                    matching["calculation"] = result.get("calculation")
                    matching["paused"] = not bool(result.get("is_terminal", True))
                    matching["status"] = "paused" if matching["paused"] else "complete"
                    matching["error"] = None
                    if not matching.get("final_answer"):
                        try:
                            state = _get_json(f"/api/v1/threads/{thread_id}/state")
                            state_values = state.get("state_values") or {}
                            matching["final_answer"] = state_values.get("final_answer")
                            matching["citations"] = state_values.get("citations") or []
                            matching["answer_status"] = state_values.get(
                                "answer_status"
                            )
                            matching["calculation"] = state_values.get("calculation")
                        except Exception:
                            pass
                    st.session_state.last_analysis = matching
                    st.success("Decision submitted and analysis state updated.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not resume this analysis: {exc}")
