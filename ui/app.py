"""Autonomous Financial Reasoning Agent - Streamlit Application.

Features:
- Tab 1: Financial Analyst Workbench (live SSE reasoning stream, scratchpad timeline, citations).
- Tab 2: Compliance HITL Inbox (active approvals, risk triggers, and parameter overrides).
"""

from __future__ import annotations

import html
import json
import uuid

import requests
import streamlit as st

from ui.sse_client import stream_sse_query

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
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --ink: #0E141F;
  --panel: #161F2E;
  --panel-2: #1C2738;
  --line: #2A3548;
  --text: #E7ECF5;
  --text-dim: #8D9AB3;
  --gold: #C9A227;
  --teal: #46B892;
  --amber: #E3A83B;
  --rose: #E2596B;
}

html, body, .stApp { background-color: var(--ink) !important; }
.stApp, .stApp p, .stApp span, .stApp li { font-family: 'IBM Plex Mono', ui-monospace, monospace; color: var(--text); }

h1, h2, h3, h4, .app-header-title, .qcard-label, .kpi-number {
  font-family: 'Space Grotesk', sans-serif !important;
  letter-spacing: -0.01em;
  color: var(--text) !important;
}

.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1280px; }

[data-testid="stHeader"] { background: var(--ink) !important; border-bottom: 1px solid var(--line); }

[data-testid="stSidebar"] { background-color: var(--panel) !important; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] * { color: var(--text) !important; }

.side-rule { border: none; border-top: 1px solid var(--line); margin: 0.75rem 0 1rem 0; }

.cap-chip-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
.cap-chip {
  font-size: 0.72rem; padding: 0.3rem 0.6rem; border: 1px solid var(--line);
  border-radius: 999px; background: var(--panel-2); color: var(--text-dim);
}

.app-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 1rem 1.25rem; margin-bottom: 1.25rem;
  background: var(--panel); border: 1px solid var(--line);
  border-top: 3px solid var(--gold); border-radius: 10px;
}
.app-header-brand { display: flex; align-items: center; gap: 0.75rem; }
.app-header-mark { font-size: 1.6rem; }
.app-header-title { font-size: 1.25rem; font-weight: 600; line-height: 1.2; }
.app-header-sub { font-size: 0.8rem; color: var(--text-dim); margin-top: 0.15rem; }
.app-header-status {
  display: flex; align-items: center; gap: 0.5rem; font-size: 0.78rem;
  padding: 0.35rem 0.75rem; border-radius: 999px; border: 1px solid var(--line);
  background: var(--panel-2); white-space: nowrap;
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--rose); animation: pulse 2s ease-in-out infinite; }
.status-teal .status-dot { background: var(--teal); }
.status-rose .status-dot { background: var(--rose); }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

.session-meta { font-size: 0.82rem; color: var(--text-dim); margin: 0.25rem 0 0.75rem 0; }
.session-meta code { background: var(--panel-2); padding: 0.1rem 0.4rem; border-radius: 4px; color: var(--gold); }

.qcard-label { font-weight: 600; font-size: 0.95rem; margin-bottom: 0.15rem; }

.badge {
  display: inline-block; font-size: 0.72rem; padding: 0.2rem 0.55rem; border-radius: 999px;
  border: 1px solid transparent; margin: 0.1rem 0.3rem 0.1rem 0; font-family: 'IBM Plex Mono', monospace;
}
.badge-rose  { background: rgba(226,89,107,0.12); color: var(--rose);  border-color: rgba(226,89,107,0.35); }
.badge-amber { background: rgba(227,168,59,0.12); color: var(--amber); border-color: rgba(227,168,59,0.35); }
.badge-teal  { background: rgba(70,184,146,0.12); color: var(--teal);  border-color: rgba(70,184,146,0.35); }
.badge-gold  { background: rgba(201,162,39,0.12); color: var(--gold);  border-color: rgba(201,162,39,0.35); }
.badge-row { margin: 0.35rem 0 0.75rem 0; }

.kpi { padding: 0.25rem 0 0.5rem 0; }
.kpi-number { font-size: 2.6rem; font-weight: 700; color: var(--gold); line-height: 1; }
.kpi-label { font-size: 0.8rem; color: var(--text-dim); margin-top: 0.2rem; }

.tl-item { display: flex; align-items: flex-start; gap: 0.6rem; padding: 0.4rem 0 0.4rem 0.75rem; border-left: 2px solid var(--line); margin-left: 0.2rem; }
.tl-dot { width: 8px; height: 8px; border-radius: 50%; margin-top: 0.35rem; flex-shrink: 0; background: var(--text-dim); }
.tl-plan .tl-dot { background: var(--gold); }
.tl-node .tl-dot { background: var(--teal); }
.tl-interrupt .tl-dot { background: var(--amber); }
.tl-error .tl-dot { background: var(--rose); }
.tl-body { display: flex; flex-direction: column; font-size: 0.85rem; }
.tl-label { font-weight: 600; color: var(--text); }
.tl-text { color: var(--text-dim); }

[data-testid="stVerticalBlockBorderWrapper"] { background: var(--panel) !important; border: 1px solid var(--line) !important; border-radius: 10px !important; }
[data-testid="stForm"] { background: var(--panel); border: 1px solid var(--line); border-top: 3px solid var(--gold); border-radius: 10px; padding: 1.25rem; }
[data-testid="stExpander"] { background: var(--panel); border: 1px solid var(--line) !important; border-radius: 10px; }

.stTabs [data-baseweb="tab-list"] { gap: 1.5rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-family: 'Space Grotesk', sans-serif; color: var(--text-dim); font-size: 0.95rem; padding-bottom: 0.6rem; }
.stTabs [aria-selected="true"] { color: var(--gold) !important; border-bottom: 2px solid var(--gold) !important; }

.stButton > button {
  border-radius: 6px; border: 1px solid var(--line); background: var(--panel-2); color: var(--text);
  font-family: 'IBM Plex Mono', monospace; transition: transform 0.12s ease, border-color 0.12s ease;
}
.stButton > button:hover { border-color: var(--gold); transform: translateY(-1px); }
.stButton > button[kind="primary"] { background: var(--gold); color: var(--ink); border: none; font-weight: 600; }
.stButton > button[kind="primary"]:hover { background: #DDB749; }

.stTextInput input, .stTextArea textarea {
  background: var(--panel-2) !important; color: var(--text) !important; border: 1px solid var(--line) !important;
  border-radius: 6px !important; font-family: 'IBM Plex Mono', monospace !important;
}

div[role="radiogroup"] { gap: 0.5rem; }
div[role="radiogroup"] label { border: 1px solid var(--line); padding: 0.3rem 0.85rem; border-radius: 6px; background: var(--panel-2); }

[data-testid="stAlert"] { background: var(--panel-2) !important; border: 1px solid var(--line) !important; border-radius: 8px !important; }
[data-testid="stCodeBlock"] pre { background: var(--ink) !important; border: 1px solid var(--line) !important; border-radius: 6px !important; }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--ink); }
::-webkit-scrollbar-thumb { background: var(--line); border-radius: 5px; }
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
        r = requests.get(url.rstrip("/") + "/", timeout=1.5)
        return r.status_code < 500
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------
st.sidebar.title("Gateway config")
api_base_url = st.sidebar.text_input("Backend API base URL", value="http://127.0.0.1:8000")
st.sidebar.markdown('<hr class="side-rule" />', unsafe_allow_html=True)
st.sidebar.markdown("**Active capabilities**")
_capabilities = [
    "Multi-modal retrieval (Neo4j + pgvector)",
    "AST dynamic fan-out (Send API)",
    "Deterministic financial risk engine",
    "Ephemeral code sandbox (Phase 9)",
]
st.sidebar.markdown(
    '<div class="cap-chip-row">' + "".join(f'<span class="cap-chip">{html.escape(c)}</span>' for c in _capabilities) + "</div>",
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
    st.caption("Ask questions across 10-K tables, MD&A commentary, and numerical calculations.")

    preset_query = ""
    preset_company = "AMZN"

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown('<div class="qcard-label">Revenue trend</div>', unsafe_allow_html=True)
            st.caption("Amazon: year-over-year revenue change, 2019 to 2020")
            if st.button("Run this query", key="preset_amzn_rev", use_container_width=True):
                preset_query = "What was the percentage change in Amazon operating revenue from 2019 to 2020?"
                preset_company = "AMZN"

    with col2:
        with st.container(border=True):
            st.markdown('<div class="qcard-label">Cloud margin drivers</div>', unsafe_allow_html=True)
            st.caption("Microsoft: what drove commercial cloud margin expansion in 2021")
            if st.button("Run this query", key="preset_msft_cloud", use_container_width=True):
                preset_query = "What drove the commercial cloud margin expansion for Microsoft in 2021?"
                preset_company = "MSFT"

    with col3:
        with st.container(border=True):
            st.markdown(
                f'<div class="qcard-label">Outlier check {badge("Triggers review", "rose")}</div>',
                unsafe_allow_html=True,
            )
            st.caption("Verify a flagged 145% operating margin and goodwill impairment")
            if st.button("Run this query", key="preset_outlier", use_container_width=True):
                preset_query = "Verify the 145% operating margin and goodwill impairment for acquisition."
                preset_company = "AMZN"

    # Query Input Form
    with st.form("query_form", clear_on_submit=False):
        c1, c2 = st.columns([4, 1])
        user_query = c1.text_area(
            "Natural language financial query",
            value=preset_query or "What was the growth in Amazon revenue from 2019 to 2020?",
            height=85,
        )
        company_id = c2.text_input("Ticker / ID", value=preset_company)
        submitted = st.form_submit_button("Run analysis", use_container_width=True, type="primary")

    if submitted and user_query.strip():
        thread_id = f"ui_thread_{uuid.uuid4().hex[:8]}"
        st.markdown(
            f'<div class="session-meta">Session thread <code>{html.escape(thread_id)}</code> '
            f'for <strong>{html.escape(company_id)}</strong></div>',
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown("#### Reasoning trace")

            timeline_logs = []
            final_answer_text = None
            is_paused_for_hitl = False

            payload = {
                "query": user_query,
                "company_identifier": company_id,
                "thread_id": thread_id,
            }

            with st.status("Initializing reasoning trajectory…", expanded=True) as status_box:
                try:
                    for event_envelope in stream_sse_query(api_base_url, payload):
                        event_type = event_envelope.get("event")
                        data = event_envelope.get("data", {})

                        if event_type == "lifecycle":
                            status_box.update(label="Planning execution strategy…")
                            st.markdown(
                                timeline_item("plan", "Planner", "Decomposed the question into an execution graph."),
                                unsafe_allow_html=True,
                            )

                        elif event_type == "node_update":
                            node_name = data.get("node", "unknown")
                            latest_log = data.get("latest_log", "")
                            status_box.update(label=f"Executing node: {node_name}")
                            st.markdown(
                                timeline_item("node", node_name, latest_log or "Step completed."),
                                unsafe_allow_html=True,
                            )

                        elif event_type == "interrupt":
                            is_paused_for_hitl = True
                            status_box.update(label="Paused for compliance review", state="error")
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
                            final_answer_text = data.get("final_answer")

                        elif event_type == "complete":
                            if not is_paused_for_hitl:
                                status_box.update(label="Analysis complete", state="complete")

                        elif event_type == "error":
                            status_box.update(label="Execution error encountered", state="error")
                            st.markdown(
                                timeline_item("error", data.get("error_type", "Error"), data.get("message", "")),
                                unsafe_allow_html=True,
                            )

                except Exception as ex:
                    status_box.update(label="Failed to connect to backend", state="error")
                    st.markdown(timeline_item("error", "Connection failure", str(ex)), unsafe_allow_html=True)

        # Render Final Synthesized Answer
        if final_answer_text:
            with st.container(border=True):
                st.markdown("#### Synthesized analysis")
                st.markdown(final_answer_text)


# =====================================================================
# TAB 2: Compliance HITL Approval Inbox
# =====================================================================
with tab_compliance:
    st.markdown("### Compliance & risk review queue")
    st.caption("Inspect state snapshots paused by the Phase 8 risk engine. Approve, reject, or inject overrides.")

    # Fetch Pending Approvals from FastAPI
    approvals_endpoint = f"{api_base_url.rstrip('/')}/api/v1/approvals"
    try:
        resp = requests.get(approvals_endpoint, timeout=5.0)
        resp.raise_for_status()
        approvals_data = resp.json()
        pending_items = approvals_data.get("items", [])
        fetch_error = None
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
        st.error(f"Unable to fetch approvals queue from `{approvals_endpoint}`: {fetch_error}")

    if not pending_items and not fetch_error:
        st.success("Zero pending approvals. All autonomous agent tasks are within safe risk parameters.")
    elif pending_items:
        st.warning(f"{len(pending_items)} task(s) currently paused awaiting human review.")

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
                        flags_html = "".join(badge(reason, "rose") for reason in item["trigger_reasons"])
                        st.markdown(f'<div class="badge-row">{flags_html}</div>', unsafe_allow_html=True)
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
                        "Analyst ID", value="analyst_compliance", key=f"analyst_{item['thread_id']}"
                    )
                    feedback_notes = st.text_area(
                        "Review feedback / rationale",
                        value="Reviewed 10-K footnote and verified figure.",
                        key=f"fb_{item['thread_id']}",
                    )

                    override_payload: dict = {}
                    if decision_action == "OVERRIDE":
                        override_raw = st.text_area(
                            "Override variables (JSON)",
                            value='{"corrected_amount": 14.5, "note": "Adjusted scale from % to ratio"}',
                            key=f"override_{item['thread_id']}",
                        )
                        try:
                            override_payload = json.loads(override_raw)
                        except Exception:
                            st.error("Invalid JSON format in override field.")

                    if st.button(
                        f"Submit {decision_action.title()}",
                        key=f"btn_{item['thread_id']}",
                        type="primary",
                        use_container_width=True,
                    ):
                        resume_url = f"{api_base_url.rstrip('/')}/api/v1/threads/{item['thread_id']}/resume"
                        post_data = {
                            "action": decision_action,
                            "analyst_id": analyst_name,
                            "feedback": feedback_notes,
                            "overrides": override_payload,
                        }

                        with st.spinner("Submitting decision and resuming state machine…"):
                            try:
                                r_res = requests.post(resume_url, json=post_data, timeout=30.0)
                                r_res.raise_for_status()
                                st.success(f"Thread {item['thread_id']} resumed successfully.")
                                st.rerun()
                            except Exception as ex:
                                st.error(f"Failed to resume thread: {str(ex)}")