# Financial Agentic System

Autonomous financial reasoning agent: FinQA data engineering, Neo4j knowledge graph, pgvector semantic store, LangGraph orchestration, parallel retrieval, HITL governance, sandboxed execution, evaluation harness, FastAPI gateway, and Streamlit UI.

An IEEE-style technical report describing the full architecture, pipelines, and evaluation is in [`docs/paper/main.tex`](docs/paper/main.tex) (compilable on Overleaf; TikZ diagrams included, no external images required).

## Architecture at a Glance

```
FinQA raw -> parser/normalizer -> Neo4j graph (9.2k reports, 111k values)
                                -> pgvector chunks (109k embeddings)
LangGraph: read_memory -> plan -> fan-out (graph|vector|table) -> aggregate
           -> fuse_context -> eval_risk -> [HITL gate] -> compute -> synthesize
           -> write_memory; checkpointer: Redis (durable) w/ MemorySaver fallback
Serving: FastAPI gateway (SSE streaming, async jobs, approvals queue)
         + Streamlit UI (query workbench, reasoning trace, compliance inbox)
Infra: docker-compose (neo4j, pgvector, redis, mlflow, api, ui, mcp)
```

## Key Design Decisions

- **Graph + vector hybrid retrieval.** Structured metrics live in Neo4j; narrative evidence lives in pgvector. The planner schedules both; fusion and relevance ranking happen at synthesis time. See [docs/GRAPH_SCHEMA.md](docs/GRAPH_SCHEMA.md).
- **LangGraph over a bare loop.** Typed state, conditional edges, Send-API fan-out for parallel tool execution, and native interrupts give HITL pause/resume semantics for free.
- **Company-scoped retrieval.** Vector search is filtered to one company's filings (`record_id` prefix) to prevent cross-company narrative pollution; empty filing lists stay empty (no unscoped fallback).
- **Validator-corrector synthesis.** Every numeral in the synthesized answer must exist in retrieved evidence; ungrounded figures reject the LLM draft and fall back to a deterministic template (no hallucination by contract).
- **Persistent audit ledger.** Merkle-chained, Postgres-backed, INSERT-only app role; degrades to in-memory loudly, never silently.
- **Bounded self-correction.** The synthesizer critique loop is capped at 3 iterations before declaring insufficient evidence.

## Results (dev-100 benchmark)

| Metric | Value |
|---|---|
| Execution accuracy (Acc_exe) | 52.0% |
| Program accuracy (Acc_prog) | 32.0% |
| Recall@5 | 89.4% |
| Precision@5 | 57.8% |
| MRR | 0.764 |
| NDCG@5 | 0.762 |
| Mean latency | 11.5 s (P95 20.6 s) |

Full causal-failure taxonomy and per-record states: [eval_results/benchmark_report.md](eval_results/benchmark_report.md).

## Quickstart (local)

```bash
# 1. Data pipeline (raw FinQA -> parsed -> normalized)
python evaluation/profile_finqa_dataset.py
python ingestion/parser.py
python ingestion/normalizer.py

# 2. Knowledge stores (docker compose up -d first)
python ingestion/graph_loader.py --uri bolt://localhost:7687 --user neo4j --password password --database neo4j
python knowledge/vector_store/loader.py

# 3. Gateway + UI
uvicorn agent.server.app:app --port 8000        # FastAPI (SSE, jobs, approvals)
streamlit run ui/app.py --server.port 8501      # Streamlit UI
```

Full container stack: `docker compose up -d --build` (neo4j, pgvector, redis, mlflow, api, ui, mcp).

## Repository Layout

| Path | Purpose |
|---|---|
| `agent/graph.py` | LangGraph orchestration (parallel + HITL wiring) |
| `agent/nodes/` | Plan, parallel fan-out, compute, synth, memory, HITL nodes |
| `agent/tools/` | graph/vector/table/safe-math/sandbox/memory tools + schemas |
| `agent/guardrails/` | Risk engine + Merkle audit ledger (persistent) |
| `agent/server/` | FastAPI gateway: SSE, async jobs, state, resume |
| `agent/sandbox/` | AST prefilter + subprocess/Docker runner |
| `agent/mcp/` | MCP tool/resource server + client bridge |
| `ingestion/`, `knowledge/` | FinQA parsing, graph loader, vector store |
| `evaluation/` | Benchmark runner, metrics, regression gate |
| `ui/` | Streamlit workbench + SSE client |
| `docs/` | ADRs, graph schema, paper (`docs/paper/main.tex`) |
| `eval_results/` | Benchmark reports + recorded states |

## Security Posture

- CORS allowlist (env-tunable), no wildcard origins
- Audit ledger DB role INSERT/SELECT only
- AST security prefilter blocks forbidden imports/builtins before sandbox exec
- Known TODOs (Phase 15/17): API auth + rate limiting on `/query`

## Testing

```bash
pytest tests -q            # offline suite; integration tests skip without DBs
FINAGENT_INTEGRATION_TESTS=1 pytest tests -q   # with live containers
```

## License

See LICENSE.
