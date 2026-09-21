# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Autonomous financial reasoning agent over the FinQA dataset: hybrid retrieval (Neo4j knowledge graph + pgvector vectors), a single LangGraph `StateGraph` orchestrator with parallel fan-out, HITL governance gates, sandboxed execution, MCP tool exposure, an evaluation harness, and a FastAPI + Streamlit frontend.

`project_goals.md` is the master plan (21 phases, v3) — each phase maps to a top-level directory and phases build strictly on each other. Phase status checkboxes there reflect actual progress.

## Commands

Python 3.11, virtualenv at `.venv/` (Windows: `.venv/Scripts/python.exe`).

```bash
# Install
pip install -e ".[dev]"

# Lint / type-check / format (pre-commit runs black, ruff, isort, mypy)
ruff check .
mypy agent/
black .
pre-commit run --all-files

# Tests (offline-safe by default — see "Testing" below)
pytest tests/ -v
pytest tests/test_agent_tools.py -v                    # single file
pytest tests/test_agent_tools.py::test_name -v         # single test
pytest tests/ -m "not integration"                     # integration tests need live DBs

# Local services (Neo4j, Redis/RedisStack, pgvector, MLflow, API, UI)
docker compose up -d

# API (both entry points serve the SAME app — api/main.py re-exports agent/server/app.py)
uvicorn agent.server.app:app --port 8000
streamlit run ui/app.py                                # UI on :8501
```

### Data pipeline (FinQA)

```bash
python evaluation/profile_finqa_dataset.py   # dataset profile
python ingestion/parser.py                   # raw -> data/processed/parsed/*.jsonl
python ingestion/normalizer.py               # -> data/processed/normalized/*.jsonl
python ingestion/graph_loader.py --dry-run --limit-per-split 100   # live load: add --uri/--user/--password
```

### Evaluation

```bash
python evaluation/runner.py --limit 25       # benchmark run; records per-state fixtures to eval_results/states/
python evaluation/run_regression.py --min-accuracy 0.70   # replays recorded fixtures only
```

`run_regression.py` replays `eval_results/states/*.json` fixtures and must NEVER fall back to a live run (CI has no databases). Regenerate fixtures locally with `runner.py`.

## Architecture

### One graph, progressively enhanced

The core design rule (from `project_goals.md` Design Principles): there is exactly ONE LangGraph orchestrator — `build_financial_agent_graph()` in `agent/graph.py`. Parallelization, HITL, and sandboxing are nodes/edges/interrupts on that same graph, not separate systems.

Execution flow: `read_memory → plan → (Send fan-out to sub_task_worker) → aggregate_sub_tasks → fuse_context → eval_risk → hitl_gate → apply_override → compute → synthesize_answer → write_memory`, with conditional routers (`route_after_*`) deciding between branches. Key routing subtleties:

- After a reviewer **EDIT**, pending edited retrieval calls re-enter the fan-out via `fan_out_router_node` (a plain node wrapper, because a conditional edge cannot target another conditional edge) before re-passing risk governance.
- `route_after_compute` routes post-math results back through `eval_risk` exactly once per iteration (the `risk_evaluated` flag prevents loops).

### State & memory

`AgentStateV1` (`agent/state/schema.py`) is a strict Pydantic model — the typed contract every node reads/writes. Reducer-annotated fields (`append_scratchpad`, `append_tool_results`, `merge_sub_task_results`) make the Phase 7 parallel fan-in thread-safe. Memory tiers live in `agent/memory/`: working (Redis-backed LangGraph checkpointer — also what makes HITL pause/resume durable across restarts), episodic (Postgres), procedural, and a promotion policy (`promoter.py`).

### HITL governance

`agent/guardrails/risk_engine.py` evaluates retrieved/computed figures and may set `hitl_status: PENDING`, pausing the graph at `hitl_gate`. The `/approvals` + `/threads/{thread_id}/resume` API endpoints and the Streamlit "Compliance HITL Inbox" resolve decisions (`APPROVE`/`REJECT`/`EDIT` → `apply_override_node`). Every decision is audit-logged (`agent/guardrails/persistent_audit.py`).

### Tools & prompts

Tools (`agent/tools/`) are plain typed functions — they know nothing about LangGraph. Each has an explicit JSON Schema in `agent/tools/schemas/` (required for the MCP server in `agent/mcp/server.py`) and latency/cost logging via `agent/telemetry/`. Generated code/Cypher goes through the sandbox interface (`agent/sandbox/` — prefilter, protocol, Docker-isolated runner).

All prompts are versioned files in `agent/prompts/` (`planner_v2.txt`, `synthesizer_v2.txt`, etc.) — never inline prompt strings in node code.

### LLM access

`agent/llm.py` is the single access point for every LLM call. Missing/placeholder API keys return `None` and callers fall back to deterministic rule-based planner/synthesizer paths — UNAVAILABLE is not an error. Supports OpenAI, OpenRouter (auto base-url), and custom gateways via env vars. This degradation contract is what keeps tests and CI runnable offline.

### API/UI

`agent/server/app.py` is the production gateway (FastAPI, SSE streaming, CORS allowlist); `api/main.py` re-exports it so both uvicorn targets work. Endpoints: `/query/stream`, `/query/async`, `/jobs/{job_id}`, `/approvals`, `/threads/{thread_id}/state`, `/threads/{thread_id}/resume`, `/health`. The Streamlit UI (`ui/app.py`) has a live SSE workbench tab and a HITL inbox tab, using `ui/sse_client.py`.

## Testing

`tests/conftest.py` blanks `OPENAI_API_KEY`/`OPENROUTER_API_KEY` on every test run so tests are deterministic and free — a `.env` with a real key on a dev machine would otherwise cause paid live calls and flaky e2e tests. To deliberately test with a live LLM: `FINAGENT_TESTS_LLM=1 pytest ...`.

Tests marked `pytest.mark.integration` (e.g. `tests/test_phase3_integrity.py`) require running databases (docker compose). `pytest.ini_options` sets `asyncio_mode = auto`.

## Conventions & gotchas

- **Ruff ignores E501 deliberately** (line-length 100 configured but not gated): the codebase legitimately carries long Streamlit CSS blocks, Cypher templates, and compiled regexes; automated rewrapping risks semantic drift. All other E/F/W/I/N/UP rules are enforced.
- **Docker service hostnames**: `.env` uses `localhost` (host-oriented), but `docker-compose.yml` overrides with service names (`NEO4J_URI=bolt://neo4j:7687`, `POSTGRES_HOST=pgvector`, `REDIS_URL=redis://redis:6379/0`) — `environment:` wins over `env_file:` in compose. LangSmith EU endpoint is set there too (US endpoint 403s for the EU PAT).
- **Redis must be RedisStack** (`redis/redis-stack-server` image) — the LangGraph checkpointer needs RediSearch/ReJSON modules.
- Secrets: `.env` is local-only (gitignored); `.env.example` is the committed template. Placeholder key values (`sk-...`, `changeme`) are treated as absent by `agent/llm.py`.
- Data is DVC-tracked (`data.dvc`); benchmark fixtures under `eval_results/states/` are git-tracked CI inputs.
