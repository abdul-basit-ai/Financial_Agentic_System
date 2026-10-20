# Financial Reasoning UI

Streamlit workbench for live financial analysis and human-in-the-loop compliance review.

## Run locally

Start the API first, then launch the UI from the repository root:

```bash
uvicorn agent.server.app:app --port 8000
streamlit run ui/app.py --server.port 8501
```

The UI uses `http://127.0.0.1:8000` by default. Set `API_URL` or edit the
sidebar field when the gateway is hosted elsewhere. Docker Compose injects
`http://api:8000` automatically.

## Main flows

- Analyst workbench with example queries, live SSE execution trace, and a
  persisted latest result across Streamlit reruns.
- Compliance inbox for approving, rejecting, or editing paused tool calls.
- Backend reachability indicator and explicit connection/error states.

The interface is intentionally a thin client: reasoning, state, validation,
and governance remain in the FastAPI/LangGraph backend.
