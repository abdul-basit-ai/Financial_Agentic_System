"""A single financial research workspace with contextual human review."""

from __future__ import annotations

import html
import json
import os
import uuid
from pathlib import Path
from typing import Any

import requests
import streamlit as st

try:
    from ui.sse_client import stream_sse_query
except ModuleNotFoundError:  # Streamlit may execute ui/app.py as a bare script.
    from sse_client import stream_sse_query  # type: ignore[import-not-found,no-redef]

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Ledger | Financial research",
    page_icon=":material/query_stats:",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Small render helpers
# ---------------------------------------------------------------------------
def badge(text: str, kind: str = "gold") -> str:
    return f'<span class="badge badge-{kind}">{html.escape(str(text))}</span>'


def timeline_item(kind: str, label: str, text: str = "") -> str:
    safe_label = html.escape(str(label))
    body = f'<span class="tl-label">{safe_label}</span>'
    if text:
        body += f'<span class="tl-text">{html.escape(str(text))}</span>'
    return f'<div class="tl-item tl-{kind}"><span class="tl-dot"></span><div class="tl-body">{body}</div></div>'


@st.cache_data(ttl=15, show_spinner=False)
def _backend_is_live(url: str) -> bool:
    try:
        r = requests.get(url.rstrip("/") + "/api/v1/health", timeout=1.5)
        return 200 <= r.status_code < 300
    except Exception:
        return False


_DEFAULT_QUERY = (
    "What is the percent change in annual long-term debt maturities "
    "from 2016 to 2017?"
)

if "query_input" not in st.session_state:
    st.session_state.query_input = _DEFAULT_QUERY
if "company_input" not in st.session_state:
    st.session_state.company_input = "ETR"
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None


def _select_example(query: str, company: str) -> None:
    """Load an example into the form without relying on widget defaults."""
    st.session_state.query_input = query
    st.session_state.company_input = company


# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------
# Default differs by runtime: inside the API docker network the gateway is
# reachable as http://api:8000 (127.0.0.1 inside the UI container is its own
# loopback, NOT the gateway). When you run Streamlit locally in a browser,
# point it at http://127.0.0.1:8000.

# Use the actual API_URL value when set (docker-compose injects
# http://api:8000); operators can override it with any reachable URL.
_DEFAULT_API_URL = os.getenv("API_URL") or "http://127.0.0.1:8000"

st.sidebar.markdown(
    '<div class="sidebar-brand">&#9672; &nbsp; ledger<span>RESEARCH WORKSPACE</span></div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    '<div class="sidebar-active">&#9673; &nbsp; Financial research</div>',
    unsafe_allow_html=True,
)
st.sidebar.caption("From question to evidence.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Your workflow**")
st.sidebar.markdown(
    "1. Ask a financial question\n2. Follow the evidence\n3. Review flagged decisions"
)
with st.sidebar.expander("Connection settings"):
    api_base_url = st.text_input("Backend API base URL", value=_DEFAULT_API_URL)
st.sidebar.markdown(
    '<div class="sidebar-note">BUILT FOR CAREFUL ANALYSIS<br>Filing evidence. Transparent calculations. Human oversight.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
backend_live = _backend_is_live(api_base_url) if api_base_url else False
status_label = "Backend online" if backend_live else "Backend unreachable"
status_kind = "teal" if backend_live else "rose"

st.markdown(
    f'<div class="topbar"><span>WORKSPACE / FINANCIAL RESEARCH</span>'
    f'<span class="connection status-{status_kind}"><i></i>{html.escape(status_label)}</span></div>'
    '<div class="hero"><div class="eyebrow">THE RESEARCH DESK</div>'
    "<h1>Clarity behind the numbers.</h1>"
    "<p>Explore company filings, trace the calculations, and build an evidence-backed view.</p></div>",
    unsafe_allow_html=True,
)

research, review = st.columns([2.3, 1], gap="large")

with research:
    st.markdown("### Start with a question")
    st.caption("Choose a company and tell us what you want to understand.")

    # Query Input Form
    with st.form("query_form", clear_on_submit=False):
        c1, c2 = st.columns([4, 1])
        user_query = c1.text_area(
            "Your question",
            height=130,
            key="query_input",
        )
        company_id = c2.text_input("Company ticker", key="company_input")
        submitted = st.form_submit_button(
            "Run analysis", use_container_width=True, type="primary"
        )

    with st.expander("Need a starting point? Explore example questions"):
        col1, col2, col3 = st.columns(3)

        with col1:
            with st.container(border=True):
                st.markdown(
                    '<div class="qcard-label">Debt maturity trend</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Entergy (ETR): percent change in annual long-term debt maturities, 2016 to 2017"
                )
                st.button(
                    "Use example",
                    key="preset_etr_debt",
                    use_container_width=True,
                    on_click=_select_example,
                    args=(_DEFAULT_QUERY, "ETR"),
                )

        with col2:
            with st.container(border=True):
                st.markdown(
                    '<div class="qcard-label">Driver attribution</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Entergy (ETR): share of the 2007-2008 net revenue change due to rider revenue"
                )
                st.button(
                    "Use example",
                    key="preset_etr_rider",
                    use_container_width=True,
                    on_click=_select_example,
                    args=(
                        "What percent of the change between net revenue in 2007 and 2008 was due to rider revenue?",
                        "ETR",
                    ),
                )

        with col3:
            with st.container(border=True):
                st.markdown(
                    f'<div class="qcard-label">No-data robustness {badge("Honest refusal", "rose")}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Amazon is not in this FinQA sample — the agent must report "
                    "insufficient evidence instead of fabricating figures."
                )
                st.button(
                    "Use example",
                    key="preset_outlier",
                    use_container_width=True,
                    on_click=_select_example,
                    args=(
                        "Verify the 145% operating margin and goodwill impairment for acquisition.",
                        "AMZN",
                    ),
                )

    if submitted and not user_query.strip():
        st.warning("Enter a financial question before running the analysis.")

    if submitted and user_query.strip():
        thread_id = f"ui_thread_{uuid.uuid4().hex[:8]}"

        with st.container(border=True):
            st.markdown("#### Reasoning trace")

            timeline_logs: list[dict[str, str]] = []
            final_answer_text: str | None = None
            is_paused_for_hitl = False
            run_error: str | None = None

            payload: dict[str, Any] = {
                "query": user_query,
                "company_identifier": company_id,
                "thread_id": thread_id,
            }

            with st.status(
                "Initializing reasoning trajectory…", expanded=True
            ) as status_box:
                try:
                    for event_envelope in stream_sse_query(api_base_url, payload):
                        event_type = event_envelope.get("event")
                        data = event_envelope.get("data", {})

                        if event_type == "lifecycle":
                            timeline_logs.append(
                                {
                                    "kind": "plan",
                                    "label": "Planner",
                                    "text": "Decomposed the question into an execution graph.",
                                }
                            )
                            status_box.update(label="Planning execution strategy…")
                            st.markdown(
                                timeline_item(
                                    "plan",
                                    "Planner",
                                    "Decomposed the question into an execution graph.",
                                ),
                                unsafe_allow_html=True,
                            )

                        elif event_type == "node_update":
                            node_name = str(data.get("node", "unknown"))
                            latest_log = str(data.get("latest_log", ""))
                            timeline_logs.append(
                                {
                                    "kind": "node",
                                    "label": node_name,
                                    "text": latest_log or "Step completed.",
                                }
                            )
                            status_box.update(label=f"Executing node: {node_name}")
                            st.markdown(
                                timeline_item(
                                    "node", node_name, latest_log or "Step completed."
                                ),
                                unsafe_allow_html=True,
                            )

                        elif event_type == "interrupt":
                            is_paused_for_hitl = True
                            timeline_logs.append(
                                {
                                    "kind": "interrupt",
                                    "label": "Compliance hold",
                                    "text": "Analysis paused for human review.",
                                }
                            )
                            status_box.update(
                                label="Paused for compliance review", state="error"
                            )
                            st.markdown(
                                timeline_item(
                                    "interrupt",
                                    "Compliance hold",
                                    f"Analysis paused at {data.get('paused_at_nodes')}. "
                                    "Review the flagged decision in the review panel to continue.",
                                ),
                                unsafe_allow_html=True,
                            )

                        elif event_type == "final_answer":
                            answer = data.get("final_answer")
                            final_answer_text = str(answer) if answer else None

                        elif event_type == "complete":
                            if not is_paused_for_hitl:
                                status_box.update(
                                    label="Analysis complete", state="complete"
                                )

                        elif event_type == "error":
                            run_error = str(data.get("message", "Execution failed."))
                            status_box.update(
                                label="Execution error encountered", state="error"
                            )
                            st.markdown(
                                timeline_item(
                                    "error",
                                    data.get("error_type", "Error"),
                                    data.get("message", ""),
                                ),
                                unsafe_allow_html=True,
                            )

                except Exception as ex:
                    run_error = str(ex)
                    status_box.update(
                        label="Failed to connect to backend", state="error"
                    )
                    st.markdown(
                        timeline_item("error", "Connection failure", str(ex)),
                        unsafe_allow_html=True,
                    )

        st.session_state.last_analysis = {
            "thread_id": thread_id,
            "company": company_id,
            "query": user_query,
            "final_answer": final_answer_text,
            "paused": is_paused_for_hitl,
            "error": run_error,
            "timeline": timeline_logs,
        }

    last_analysis = st.session_state.last_analysis
    if not last_analysis:
        st.markdown(
            '<div class="empty-state"><div class="empty-mark">&#8599;</div>'
            "<h3>A good question is the beginning.</h3>"
            "<p>Your analysis will appear here, with the supporting evidence and a trace of every step.</p>"
            '<div class="process"><span>01 &nbsp; Retrieve evidence</span><span>02 &nbsp; Calculate</span><span>03 &nbsp; Explain</span></div></div>',
            unsafe_allow_html=True,
        )
    if isinstance(last_analysis, dict):
        st.markdown(
            f'<div class="session-meta">Latest session <code>{html.escape(str(last_analysis.get("thread_id", "unknown")))}</code> '
            f'for <strong>{html.escape(str(last_analysis.get("company") or "N/A"))}</strong></div>',
            unsafe_allow_html=True,
        )
        if last_analysis.get("final_answer"):
            with st.container(border=True):
                st.markdown("#### Synthesized analysis")
                st.markdown(str(last_analysis["final_answer"]))
        elif last_analysis.get("paused"):
            st.info(
                "This analysis needs your review. Use the review panel to continue."
            )
        elif last_analysis.get("error"):
            st.error(f"The latest analysis failed: {last_analysis['error']}")

        prior_timeline = last_analysis.get("timeline")
        if not submitted and isinstance(prior_timeline, list) and prior_timeline:
            with st.expander("Latest execution trace"):
                for entry in prior_timeline:
                    if isinstance(entry, dict):
                        st.markdown(
                            timeline_item(
                                str(entry.get("kind", "node")),
                                str(entry.get("label", "Step")),
                                str(entry.get("text", "")),
                            ),
                            unsafe_allow_html=True,
                        )


# =====================================================================
# Contextual human review
# =====================================================================
with review:
    st.markdown("### Review desk")
    st.caption(
        "Human oversight, right where you work. Review decisions that need your attention."
    )

    # Fetch Pending Approvals from FastAPI
    approvals_endpoint = f"{api_base_url.rstrip('/')}/api/v1/approvals"
    try:
        resp = requests.get(approvals_endpoint, timeout=5.0)
        resp.raise_for_status()
        approvals_data = resp.json()
        raw_items = approvals_data.get("items", [])
        pending_items: list[dict[str, Any]] = (
            [item for item in raw_items if isinstance(item, dict)]
            if isinstance(raw_items, list)
            else []
        )
        fetch_error: str | None = None
    except Exception as ex:
        pending_items = []
        fetch_error = str(ex)

    kpi_col, refresh_col = st.columns([1, 1])
    with kpi_col:
        st.markdown(
            f'<div class="kpi"><div class="kpi-number">{len(pending_items)}</div>'
            f'<div class="kpi-label">Awaiting human review</div></div>',
            unsafe_allow_html=True,
        )
    with refresh_col:
        st.write("")
        if st.button("Refresh queue", use_container_width=True):
            st.rerun()

    if fetch_error:
        st.info(
            "The review desk is offline. Check your backend connection to load pending decisions."
        )
        with st.expander("Connection details"):
            st.caption(fetch_error)

    if not pending_items and not fetch_error:
        st.success("You are all caught up. No decisions are waiting for review.")
    elif pending_items:
        st.warning(
            f"{len(pending_items)} task(s) currently paused awaiting human review."
        )

        for idx, item in enumerate(pending_items, start=1):
            with st.expander(
                f"Case #{idx}: thread {item['thread_id']} for {item['company_identifier'] or 'N/A'}",
                expanded=True,
            ):
                col_left, col_right = st.container(), st.container()

                with col_left:
                    st.markdown("**Original query**")
                    st.markdown(f"*{item['input_query']}*")
                    st.markdown(f"**Paused at nodes:** `{item['paused_nodes']}`")

                    st.markdown("**Risk engine flags**")
                    if item["trigger_reasons"]:
                        flags_html = "".join(
                            badge(reason, "rose") for reason in item["trigger_reasons"]
                        )
                        st.markdown(
                            f'<div class="badge-row">{flags_html}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="badge-row">{badge("Materiality / confidence bound", "amber")}</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown("**Recent reasoning logs**")
                    for log in item["scratchpad_summary"]:
                        st.code(log, language="text")

                with col_right:
                    st.markdown("**Analyst decision**")
                    decision_action = st.radio(
                        "Action",
                        ["APPROVE", "OVERRIDE", "REJECT"],
                        key=f"action_{item['thread_id']}",
                        horizontal=True,
                        label_visibility="collapsed",
                    )
                    analyst_name = st.text_input(
                        "Analyst ID",
                        value="analyst_compliance",
                        key=f"analyst_{item['thread_id']}",
                    )
                    feedback_notes = st.text_area(
                        "Review feedback / rationale",
                        value="",
                        placeholder="Explain the evidence behind your decision...",
                        key=f"fb_{item['thread_id']}",
                    )

                    override_payload: dict[str, Any] = {}
                    override_is_valid = True
                    if decision_action == "OVERRIDE":
                        # OVERRIDE maps to the graph's EDIT verb, which only
                        # consumes overrides.tool_calls (the replacement queue
                        # of pending tool calls). Prefill with the thread's
                        # actual pending calls so the analyst edits real data —
                        # arbitrary JSON here used to be silently ignored.
                        default_calls = item.get("pending_tool_calls") or []
                        override_raw = st.text_area(
                            "Edited tool calls (JSON list)",
                            value=json.dumps(
                                {"tool_calls": default_calls}, indent=2, default=str
                            ),
                            height=260,
                            key=f"override_{item['thread_id']}",
                        )
                        try:
                            override_payload = json.loads(override_raw)
                            edited = override_payload.get("tool_calls")
                            if not isinstance(edited, list) or not all(
                                isinstance(c, dict)
                                and "task_id" in c
                                and "target_tool" in c
                                for c in edited
                            ):
                                raise ValueError(
                                    "tool_calls must be a list of objects with "
                                    "task_id, target_tool and payload keys"
                                )
                        except ValueError as ve:
                            st.error(f"Invalid override structure: {ve}")
                            override_payload = {}
                            override_is_valid = False
                        except Exception:
                            st.error("Invalid JSON format in override field.")
                            override_payload = {}
                            override_is_valid = False

                    if st.button(
                        f"Submit {decision_action.title()}",
                        key=f"btn_{item['thread_id']}",
                        type="primary",
                        use_container_width=True,
                        disabled=not override_is_valid,
                    ):
                        resume_url = f"{api_base_url.rstrip('/')}/api/v1/threads/{item['thread_id']}/resume"
                        post_data: dict[str, Any] = {
                            "action": decision_action,
                            "analyst_id": analyst_name,
                            "feedback": feedback_notes,
                            "overrides": override_payload,
                        }

                        with st.spinner(
                            "Submitting decision and resuming state machine…"
                        ):
                            try:
                                r_res = requests.post(
                                    resume_url, json=post_data, timeout=30.0
                                )
                                r_res.raise_for_status()
                                st.success(
                                    f"Thread {item['thread_id']} resumed successfully."
                                )
                                st.rerun()
                            except Exception as ex:
                                st.error(f"Failed to resume thread: {str(ex)}")
