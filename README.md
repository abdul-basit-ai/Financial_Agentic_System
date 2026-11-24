# Financial Agentic System

A private research workspace for English-language annual reports and SEC filings.
The Streamlit UI sends authenticated requests to a FastAPI gateway. A Python
worker processes uploaded PDFs and runs durable jobs. Supabase Auth, private
Storage, and Postgres are the production data services. Local development uses a
bearer token, SQLite, and filesystem storage.

The system extracts page-level text and table evidence, uses English Tesseract
OCR for image-only pages, and returns citations with page and table coordinates.
Ambiguous evidence or units should result in an explicit insufficient-evidence
answer. Arithmetic is auditable from cited values. Arbitrary Python execution is
not part of the supported workflow.

## Architecture

```text
Streamlit UI -> authenticated FastAPI -> Supabase Auth
                                      -> Supabase Postgres
                                      -> private Supabase Storage
                                      -> Python worker
                                           -> PDF text and table extraction
                                           -> English OCR for scanned pages
                                           -> evidence-grounded answers
```

Production runs the API and worker as separate processes against shared durable
Postgres and private object storage. Local SQLite and filesystem paths are for
development only. The application does not create Supabase projects, deploy
cloud resources, or apply its Supabase application schema automatically. The
Postgres checkpointer initializes its private tables during API startup.

## Local setup

Python 3.11 is supported. Install [uv](https://docs.astral.sh/uv/) and Tesseract
with English language data. Docker images install `tesseract-ocr-eng` for you.

```bash
uv sync --extra dev
cp .env.example .env
```

Set a unique local `FINAGENT_API_TOKEN` in `.env`. Add an OpenAI-compatible
provider key if model calls are needed; the application can start without one,
but deterministic fallback behavior is more limited. Start the API:

```bash
uv run --env-file .env uvicorn agent.server.app:app --host 127.0.0.1 --port 8000
```

In a second terminal, start the UI and enter the same local token in its sidebar:

```bash
uv run streamlit run ui/app.py --server.port 8501
```

The default local API embeds the worker. To run it as a separate process, set
`FINAGENT_EMBEDDED_WORKER=0` in `.env` and start the worker in another terminal:

```bash
uv run --env-file .env python -m agent.worker
```

Do not run multiple SQLite-backed API or worker replicas against separate local
files.

## Filing research

Upload a PDF report from the filing library. The API enforces the configured
upload limit (`MAX_UPLOAD_BYTES`, 20 MiB by default) and page limit
(`MAX_DOCUMENT_PAGES`, 1,000 by default). Text-based pages use native PDF
extraction. Image-only pages use the English Tesseract data installed on the
host. Select reports after their status becomes ready, then ask a question. The
result includes its answer status, page references, available table coordinates,
quoted evidence, and calculation details. The review desk resumes analyses
that request human review.

## Supabase deployment configuration

Create a Supabase project, enable the sign-in method you intend to use, and apply
[`supabase/schema.sql`](supabase/schema.sql) manually in the SQL editor. The
schema creates the application tables and a private `financial-reports` bucket.
Review the SQL and your project policies before applying it. This application
schema is a manual step. The LangGraph Postgres checkpointer initializes its own
checkpoint tables when the API starts; that library-owned setup is separate.

Configure the server environment with:

- `FINAGENT_AUTH_MODE=supabase`
- `SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY`
- `SUPABASE_SECRET_KEY` for server-side private Storage access
- `DATABASE_URL` for the Supabase Postgres pooler
- `SUPABASE_REPORT_BUCKET=financial-reports`
- `FINAGENT_CHECKPOINT_BACKEND=postgres`
- Provider credentials and task-specific model names as needed

Keep `SUPABASE_SECRET_KEY`, `DATABASE_URL`, and model provider keys on the API
and worker only. The UI container receives only `API_URL`; it does not receive
server credentials. Never commit `.env` or place server credentials in browser
configuration. The sample `.env` is for local token authentication. Production
deployment overrides it to use Supabase Auth.

### Model routing

Set `FINAGENT_DISABLE_LLM=1` for deterministic offline checks. This prevents
provider initialization even when a local `.env` contains configured credentials.

`OPENAI_MODEL` is the fallback model name. Optional task overrides are
`PLANNER_MODEL`, `PROGRAM_MODEL`, `PROGRAM_VERIFIER_MODEL`, `SYNTHESIS_MODEL`,
`DOCUMENT_REASONING_MODEL`, and `EXTRACTION_MODEL`. Model names are configuration
placeholders; quality, latency, and cost have not been established for a
production model selection. The runtime still has a `gpt-4o-mini` compatibility
fallback when no model name is configured; this is not a recommendation or a
measured production choice. Provider calls may send report excerpts to the
configured provider. `MODEL_PRICING_JSON` can override estimated rates used for
cost reporting. Remote tracing is disabled by default; set
`FINAGENT_ENABLE_REMOTE_TRACING=1` only when you intend to send trace data to the
configured tracing service.

`DOCUMENT_OCR_DPI` controls English OCR resolution and accepts values from 160
to 360; the default is 240. `WORKER_LEASE_SECONDS` controls how long a worker
holds a claimed durable job before another worker can reclaim it.

## Docker Compose

The local stack starts the API, separate worker, and UI, using the local SQLite
and filesystem paths from `.env`:

```bash
docker compose up --build
```

Open `http://127.0.0.1:8501`. The API and UI ports bind to loopback. The UI gets
the API URL through the internal Compose network and asks for the local token.

The Supabase-backed Compose example expects the same server-only values in `.env`:

```bash
docker compose -f docker-compose.prod.yml up --build
```

It runs the API, worker, and UI image separately, uses Supabase Auth and Postgres,
and binds ports to loopback for a host reverse proxy. It does not provide
TLS, secret management, backups, monitoring, or a high-availability guarantee.
Review and operate those services separately before any production use.

## Kubernetes reference manifests

The manifests under `k8s/` describe API, worker, and UI deployments using
Supabase-backed configuration and `/ready` probes. Before applying them, install
the [External Secrets Operator](https://external-secrets.io/latest/api/externalsecret/), create a `ClusterSecretStore` named
`financial-agent-secret-store`, configure its remote secret `financial-agent`,
and customize the example ExternalSecret, image, TLS issuer, hostname, resource
limits, and namespace for your cluster. The remote secret must provide
`SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`, `SUPABASE_SECRET_KEY`, and
`DATABASE_URL`. Add `OPENAI_API_KEY` to the secret mapping only if using a model
provider.

Apply `supabase/schema.sql` and create/verify the private bucket separately,
then deploy with `kubectl apply -k k8s/overlays/prod`. These are deployment
examples, not a tested or certified production platform. Do not run the archived
AWS Terraform under `infra/archive/aws`; it is unsupported scaffolding.

## Evaluation

The checked-in historical benchmark report was produced with older scoring and
is not a valid measurement of the current candidate. Its accuracy and retrieval
figures are not presented as current system results. CI runs an offline
candidate-runtime regression suite; it does not claim full-corpus accuracy.

Run a versioned FinQA benchmark against an immutable retrieval-corpus ID and a
new output directory:

```bash
uv run python -m evaluation.runner --scope known-record --corpus-id <immutable-index-id> --output-dir <new-run-directory>
```

Use `--scope open-corpus` for an evaluation intended to measure retrieval over
an open corpus. Preserve each run's manifest and output directory. The evaluation
runner rejects missing/ambiguous record IDs and mismatched saved states. Results
from scorer-only replay are separate from candidate-runtime runs.

## Checks

```bash
uv sync --extra dev
uv run ruff check agent api ingestion knowledge evaluation observability ui infra tests
uv run black --check --target-version py311 --line-length 88 agent api ingestion knowledge evaluation observability ui infra tests
uv run mypy agent api ingestion knowledge evaluation observability ui
uv run pytest -q
uv run python -m evaluation.run_regression
uv run pip-audit
```

`/health` is a liveness endpoint. `/ready` checks that the configured data store,
authentication mode, and compiled agent are ready.

## License

No license has been selected for this repository.
