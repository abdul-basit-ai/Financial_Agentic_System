# FastAPI gateway

`api.main:app` re-exports `agent.server.app:app`. Both import paths serve the
same authenticated API and readiness checks.

For local token-authenticated development, provide `FINAGENT_API_TOKEN` and
start the app with the local environment file:

```bash
uv run --env-file .env uvicorn agent.server.app:app --host 127.0.0.1 --port 8000
```

Production mode uses `FINAGENT_AUTH_MODE=supabase`, Supabase Auth, private
Storage, and a Supabase Postgres `DATABASE_URL`. Apply `supabase/schema.sql`
manually before starting the services. The API does not apply the application
schema at startup; the LangGraph Postgres checkpointer initializes its own
checkpoint tables during API startup. Keep Supabase secret keys, database
credentials, and provider keys on the API and worker only.

`GET /health` is liveness. `GET /ready` checks the configured store,
authentication configuration, and compiled agent.

Public setup endpoints are `GET /api/v1/config`, `POST /api/v1/auth/login`, and
`POST /api/v1/auth/refresh`. Protected endpoints require a bearer token. The
document API supports listing, PDF upload, status inspection, evidence reading,
download, and deletion within the authenticated user's ownership scope.

Research and review endpoints include:

- `POST /api/v1/query/stream` for authenticated SSE analysis
- `GET /api/v1/approvals` for the owner's review queue
- `GET /api/v1/threads/{id}/state` and `POST /api/v1/threads/{id}/resume`
- `POST /api/v1/query/async` and `GET /api/v1/jobs/{id}` for durable jobs
