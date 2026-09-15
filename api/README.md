# API

This package re-exports the production gateway: `api.main:app` IS
`agent.server.app:app` (FastAPI + LangGraph agent, SSE streaming, HITL
governance endpoints). It used to be a bare `/health` stub, which made it easy
to accidentally serve an API without the agent wired in.

Run either way:

```bash
uvicorn agent.server.app:app --host 0.0.0.0 --port 8000
# or equivalently
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Key endpoints:

- `GET /health`, `GET /healthz` — liveness (used by K8s probes and UI status)
- `POST /api/v1/query/stream` — SSE reasoning stream
- `POST /api/v1/query/async` + `GET /api/v1/jobs/{id}` — background execution
- `GET /api/v1/approvals` — HITL review queue
- `GET /api/v1/threads/{id}/state`, `POST /api/v1/threads/{id}/resume` — inspect/resume paused threads
