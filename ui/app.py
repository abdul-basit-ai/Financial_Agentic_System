"""Autonomous Financial Reasoning Agent - Streamlit Application.

Features:
- Tab 1: Financial Analyst Workbench (live SSE reasoning stream, scratchpad timeline, citations).
- Tab 2: Compliance HITL Inbox (active approvals, risk triggers, and parameter overrides).
"""

from __future__ import annotations

import html
import json
import os
import uuid
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
    page_title="Financial Reasoning Gateway",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
:root {
  --canvas: #F3F6FA;
  --surface: #FFFFFF;
  --surface-soft: #F7F9FC;
  --line: #DCE3EC;
  --text: #12233F;
  --text-dim: #66758C;
  --navy: #0B1F3A;
  --blue: #2563EB;
  --teal: #0F8A72;
  --amber: #B86E00;
  --rose: #C43D58;
  --shadow: 0 16px 44px rgba(25, 45, 78, 0.08);
}

html, body, .stApp { background-color: var(--canvas) !important; }
.stApp, .stApp p, .stApp span, .stApp li {
  font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
  color: var(--text);
}

h1, h2, h3, h4, .app-header-title, .qcard-label, .kpi-number {
  font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
  letter-spacing: -0.02em;
  color: var(--text) !important;
}

.block-container { padding-top: 1.25rem; padding-bottom: 3rem; max-width: 1240px; }

[data-testid="stHeader"] { background: rgba(243, 246, 250, 0.82) !important; backdrop-filter: blur(14px); }

[data-testid="stSidebar"] { background: linear-gradient(180deg, #0B1F3A 0%, #102B4E 100%) !important; border-right: 0; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: #F7FAFF !important; }
[data-testid="stSidebar"] input { background: rgba(255,255,255,0.1) !important; color: #FFFFFF !important; border-color: rgba(255,255,255,0.22) !important; }

.side-rule { border: none; border-top: 1px solid rgba(255,255,255,0.16); margin: 0.9rem 0 1rem; }

.cap-chip-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
.cap-chip {
  font-size: 0.7rem; padding: 0.32rem 0.6rem; border: 1px solid rgba(255,255,255,0.16);
  border-radius: 999px; background: rgba(255,255,255,0.08); color: #DDE8F7 !important;
}

.app-header {
  display: flex; align-items: center; justify-content: space-between;
  gap: 1.5rem; padding: 1.4rem 1.55rem; margin-bottom: 1.35rem;
  background: linear-gradient(120deg, #0B1F3A 0%, #153E67 68%, #0F766E 130%);
  border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; box-shadow: var(--shadow);
}
.app-header-brand { display: flex; align-items: center; gap: 0.75rem; }
.app-header-mark { display: grid; place-items: center; width: 42px; height: 42px; border-radius: 12px; background: rgba(255,255,255,0.12); font-size: 1.3rem; }
.app-header-title { font-size: 1.28rem; font-weight: 720; line-height: 1.2; color: #FFFFFF !important; }
.app-header-sub { font-size: 0.82rem; color: #C9D7E9 !important; margin-top: 0.2rem; max-width: 720px; }
.app-header-status {
  display: flex; align-items: center; gap: 0.5rem; font-size: 0.78rem;
  padding: 0.42rem 0.72rem; border-radius: 999px; border: 1px solid rgba(255,255,255,0.18);
  background: rgba(255,255,255,0.1); white-space: nowrap; color: #FFFFFF !important;
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--rose); animation: pulse 2s ease-in-out infinite; }
.status-teal .status-dot { background: #39D7B5; }
.status-rose .status-dot { background: #FF718B; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

.session-meta { font-size: 0.82rem; color: var(--text-dim); margin: 0.25rem 0 0.75rem 0; }
.session-meta code { background: #E8EEF8; padding: 0.12rem 0.42rem; border-radius: 5px; color: var(--blue); }

.qcard-label { font-weight: 700; font-size: 0.95rem; margin-bottom: 0.2rem; }

.badge {
  display: inline-block; font-size: 0.72rem; padding: 0.2rem 0.55rem; border-radius: 999px;
  border: 1px solid transparent; margin: 0.1rem 0.3rem 0.1rem 0;
}
.badge-rose  { background: #FDECF0; color: var(--rose); border-color: #F7C8D2; }
.badge-amber { background: #FFF4DE; color: var(--amber); border-color: #F4D79D; }
.badge-teal  { background: #E6F7F3; color: var(--teal); border-color: #B8E5DA; }
.badge-gold  { background: #EAF0FF; color: var(--blue); border-color: #C9D8FF; }
.badge-row { margin: 0.35rem 0 0.75rem 0; }

.kpi { padding: 0.25rem 0 0.5rem 0; }
.kpi-number { font-size: 2.6rem; font-weight: 760; color: var(--blue) !important; line-height: 1; }
.kpi-label { font-size: 0.8rem; color: var(--text-dim); margin-top: 0.2rem; }

.tl-item { display: flex; align-items: flex-start; gap: 0.6rem; padding: 0.4rem 0 0.4rem 0.75rem; border-left: 2px solid var(--line); margin-left: 0.2rem; }
.tl-dot { width: 8px; height: 8px; border-radius: 50%; margin-top: 0.35rem; flex-shrink: 0; background: var(--text-dim); }
.tl-plan .tl-dot { background: var(--blue); }
.tl-node .tl-dot { background: var(--teal); }
.tl-interrupt .tl-dot { background: var(--amber); }
.tl-error .tl-dot { background: var(--rose); }
.tl-body { display: flex; flex-direction: column; font-size: 0.85rem; }
.tl-label { font-weight: 600; color: var(--text); }
.tl-text { color: var(--text-dim); }

[data-testid="stVerticalBlockBorderWrapper"] { background: var(--surface) !important; border: 1px solid var(--line) !important; border-radius: 14px !important; box-shadow: 0 8px 24px rgba(31, 50, 81, 0.045); }
[data-testid="stForm"] { background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 1.25rem; box-shadow: var(--shadow); }
[data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--line) !important; border-radius: 12px; }

.stTabs [data-baseweb="tab-list"] { gap: 1.5rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { color: var(--text-dim); font-size: 0.92rem; font-weight: 600; padding-bottom: 0.65rem; }
.stTabs [aria-selected="true"] { color: var(--blue) !important; border-bottom: 2px solid var(--blue) !important; }

.stButton > button {
  border-radius: 9px; border: 1px solid var(--line); background: var(--surface); color: var(--text);
  font-weight: 620; transition: transform 0.12s ease, border-color 0.12s ease, box-shadow 0.12s ease;
}
.stButton > button:hover { border-color: #9EB7E4; transform: translateY(-1px); box-shadow: 0 7px 18px rgba(30, 73, 145, 0.1); }
.stButton > button[kind="primary"] { background: var(--blue); color: #FFFFFF; border: none; font-weight: 680; }
.stButton > button[kind="primary"]:hover { background: #1D4ED8; }

.stTextInput input, .stTextArea textarea {
  background: var(--surface-soft) !important; color: var(--text) !important; border: 1px solid var(--line) !important;
  border-radius: 9px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus { border-color: #8BADF0 !important; box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important; }

div[role="radiogroup"] { gap: 0.5rem; }
div[role="radiogroup"] label { border: 1px solid var(--line); padding: 0.3rem 0.85rem; border-radius: 8px; background: var(--surface-soft); }

[data-testid="stAlert"] { border: 1px solid var(--line) !important; border-radius: 10px !important; }
[data-testid="stCodeBlock"] pre { background: #0F1F35 !important; border: 1px solid #203752 !important; border-radius: 8px !important; }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--canvas); }
::-webkit-scrollbar-thumb { background: var(--line); border-radius: 5px; }

@media (max-width: 760px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .app-header { align-items: flex-start; flex-direction: column; }
  .app-header-status { align-self: flex-start; }
}
"""

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
        r = requests.get(url.rstrip("/") + "/health", timeout=1.5)
        return r.status_code < 500
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

st.sidebar.title("Gateway config")
api_base_url = st.sidebar.text_input("Backend API base URL", value=_DEFAULT_API_URL)
st.sidebar.markdown('<hr class="side-rule" />', unsafe_allow_html=True)
st.sidebar.markdown("**Active capabilities**")
_capabilities = [
    "Multi-modal retrieval (Neo4j + pgvector)",
    "AST dynamic fan-out (Send API)",
    "Deterministic financial risk engine",
    "Ephemeral code sandbox (Phase 9)",
]
st.sidebar.markdown(
    '<div class="cap-chip-row">'
    + "".join(f'<span class="cap-chip">{html.escape(c)}</span>' for c in _capabilities)
    + "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
backend_live = _backend_is_live(api_base_url) if api_base_url else False
status_label = "Backend online" if backend_live else "Backend unreachable"
status_kind = "teal" if backend_live else "rose"

st.markdown(
    f"""
    <div class="app-header">
      <div class="app-header-brand">
        <span class="app-header-mark">🧮</span>
        <div>
          <div class="app-header-title">Financial Reasoning Gateway</div>
          <div class="app-header-sub">Autonomous multi-hop analysis over 10-K filings, with human-in-the-loop compliance review</div>
        </div>
      </div>
      <div class="app-header-status status-{status_kind}"><span class="status-dot"></span>{html.escape(status_label)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main navigation tabs
tab_analyst, tab_compliance = st.tabs(["📈 Analyst workbench", "🛡️ Compliance inbox"])


# =====================================================================
# TAB 1: Financial Analyst Workbench
# =====================================================================
with tab_analyst:
    st.markdown("### Multi-hop financial reasoning")
    st.caption(
        "Ask questions across 10-K tables, MD&A commentary, and numerical calculations."
    )

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

    # Query Input Form
    with st.form("query_form", clear_on_submit=False):
        c1, c2 = st.columns([4, 1])
        user_query = c1.text_area(
            "Natural language financial query",
            height=85,
            key="query_input",
        )
        company_id = c2.text_input("Ticker / ID", key="company_input")
        submitted = st.form_submit_button(
            "Run analysis", use_container_width=True, type="primary"
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
                                    f"Trajectory frozen at {data.get('paused_at_nodes')}. "
                                    "Open the Compliance inbox tab to review risk flags and resume.",
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
                "This analysis is paused for human review. Open the Compliance inbox to continue it."
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
# TAB 2: Compliance HITL Approval Inbox
# =====================================================================
with tab_compliance:
    st.markdown("### Compliance & risk review queue")
    st.caption(
        "Inspect state snapshots paused by the Phase 8 risk engine. Approve, reject, or inject overrides."
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

    kpi_col, refresh_col = st.columns([3, 1])
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
        st.error(
            f"Unable to fetch approvals queue from `{approvals_endpoint}`: {fetch_error}"
        )

    if not pending_items and not fetch_error:
        st.success(
            "Zero pending approvals. All autonomous agent tasks are within safe risk parameters."
        )
    elif pending_items:
        st.warning(
            f"{len(pending_items)} task(s) currently paused awaiting human review."
        )

        for idx, item in enumerate(pending_items, start=1):
            with st.expander(
                f"Case #{idx}: thread {item['thread_id']} for {item['company_identifier'] or 'N/A'}",
                expanded=True,
            ):
                col_left, col_right = st.columns([3, 2])

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
                        value="Reviewed 10-K footnote and verified figure.",
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
