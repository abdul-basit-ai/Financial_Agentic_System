# Financial Reasoning UI

Ledger is a single Streamlit workspace for financial research and human review.
The question composer and results sit alongside a review desk, with no navigation tabs.

## Run locally

Start the API first, then launch the UI from the repository root:

```bash
uvicorn agent.server.app:app --port 8000
streamlit run ui/app.py --server.port 8501
```

The UI uses `http://127.0.0.1:8000` by default. Set `API_URL` or edit the
sidebar's Connection settings when the gateway is hosted elsewhere. Docker Compose injects
`http://api:8000` automatically.

## Main flows

- Research desk with optional example queries, live SSE execution trace, and a
  persisted latest result across Streamlit reruns.
- Adjacent review desk for approving, rejecting, or editing paused tool calls.
- Backend reachability indicator and explicit connection/error states.

The interface is intentionally a thin client: reasoning, state, validation,
and governance remain in the FastAPI/LangGraph backend.

Visual styles live in `ui/styles.css`. On narrow screens, the review desk stacks
below the research area. Connection failures show a compact message with
technical details available on demand.
