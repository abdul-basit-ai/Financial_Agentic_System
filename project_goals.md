# Autonomous Financial Reasoning Agent — Project Plan (v3)

> **What changed from v2:** Restructured around the actual **development lifecycle of an agent** — environment → data → knowledge → tools → state schema → memory → orchestration (LangGraph) → parallelization → HITL → sandboxing → MCP → evaluation → observability → security → CI/CD → containerization → IaC → Kubernetes → MLOps → load testing → portfolio packaging. Every phase now builds strictly on artifacts produced by the phase before it — no forward references, no orphaned tasks. LangGraph is the orchestration layer throughout (state graph, not a bare while-loop). All MLOps/DevOps tooling (CI, CD, Docker, IaC, K8s, experiment tracking, secrets, load testing) is integrated at the point in the lifecycle where it naturally belongs, plus a dedicated platform phase for the cloud-native layer.

---

## Table of Contents

1. [Design Principles](#design-principles)
2. [Phase Overview](#phase-overview)
3. [Phase 1 — Environment & Project Scaffolding](#phase-1)
4. [Phase 2 — Data Engineering](#phase-2)
5. [Phase 3 — Knowledge Graph & Vector Store](#phase-3)
6. [Phase 4 — Agent Tooling](#phase-4)
7. [Phase 5 — Agent State Schema & Memory](#phase-5)
8. [Phase 6 — Orchestration with LangGraph](#phase-6)
9. [Phase 7 — Parallelization](#phase-7)
10. [Phase 8 — Human-in-the-Loop](#phase-8)
11. [Phase 9 — Sandboxed Execution](#phase-9)
12. [Phase 10 — MCP Integration](#phase-10)
13. [Phase 11 — Evaluation & Testing](#phase-11)
14. [Phase 12 — Observability](#phase-12)
15. [Phase 13 — Security & Hardening](#phase-13)
16. [Phase 14 — API & UI](#phase-14)
17. [Phase 15 — CI/CD](#phase-15)
18. [Phase 16 — Containerization](#phase-16)
19. [Phase 17 — Infrastructure as Code](#phase-17)
20. [Phase 18 — Kubernetes Deployment](#phase-18)
21. [Phase 19 — MLOps Layer](#phase-19)
22. [Phase 20 — Load & Resilience Testing](#phase-20)
23. [Phase 21 — Portfolio Packaging](#phase-21)
24. [Master Timeline](#master-timeline)
25. [MVP → MVP+ → Full Build](#mvp-path)
26. [Why This Matters](#why-this-matters)

---

## <a name="design-principles"></a>Design Principles

1. **Schema before state before memory before orchestration.** You cannot build a LangGraph graph without a typed state; you cannot build memory without knowing what belongs in state vs. what belongs in a store; you cannot orchestrate without both existing first.
2. **Tools before graph.** LangGraph nodes call tools. Tools must exist, be tested, and be latency/cost-instrumented before they're wired into a graph.
3. **Single graph, progressively enhanced.** There is one LangGraph `StateGraph` built in Phase 6. Parallelization (7), HITL (8), and sandboxing (9) are **not separate systems** — they are nodes/edges/interrupts added to that same graph. This avoids the common mistake of bolting on a second orchestration mechanism later.
4. **Local-first, cloud-later.** Everything through Phase 16 runs on `docker-compose` on a laptop. IaC/K8s (17-18) only provision cloud infra to run the same containers — they never change application logic.
5. **Every phase produces a committed artifact** (code, config, or doc) that the next phase depends on. No phase is "just planning."

---

## <a name="phase-overview"></a>Phase Overview

| Phase | Name | Duration | Priority | Depends On |
|---|---|---|---|---|
| 1 | Environment & Project Scaffolding | Week 1 | Critical | — |
| 2 | Data Engineering | Week 1-2 | Critical | 1 |
| 3 | Knowledge Graph & Vector Store | Week 2-3 | Critical | 2 |
| 4 | Agent Tooling | Week 3 | Critical | 3 |
| 5 | Agent State Schema & Memory | Week 3-4 | Critical | 4 |
| 6 | Orchestration with LangGraph | Week 4 | Critical | 5 |
| 7 | Parallelization | Week 4-5 | Important | 6 |
| 8 | Human-in-the-Loop | Week 5 | Important | 6 |
| 9 | Sandboxed Execution | Week 5 | Important | 4, 6 |
| 10 | MCP Integration | Week 5-6 | Important | 4, 6 |
| 11 | Evaluation & Testing | Week 6 | Critical | 6 |
| 12 | Observability | Week 6-7 | Important | 6-11 |
| 13 | Security & Hardening | Week 7 | Important | 12 |
| 14 | API & UI | Week 7 | Critical | 6-9 |
| 15 | CI/CD | Week 7-8 | Important | 11, 13, 14 |
| 16 | Containerization | Week 8 | Important | 14, 15 |
| 17 | Infrastructure as Code | Week 8 | Important | 16 |
| 18 | Kubernetes Deployment | Week 8-9 | Important | 16, 17 |
| 19 | MLOps Layer | Week 9 | Important | 11, 18 |
| 20 | Load & Resilience Testing | Week 9 | Important | 18 |
| 21 | Portfolio Packaging | Week 9-10 | Final | All |

**Why this order fixes the v2 issues:**
- State schema and memory (5) now come *before* orchestration (6) — you can't build LangGraph nodes around a state shape that doesn't exist yet.
- Parallelization, HITL, and sandboxing (7-9) come *after* the base graph (6) exists, since they're extensions of it, not parallel workstreams.
- MCP (10) comes after tools (4) and the graph (6), since it wraps things that must already work standalone.
- Evaluation (11) comes after the graph is orchestrating end-to-end, not before — you can't evaluate an agent that doesn't run yet.
- CI/CD (15) comes after there's something to test and build (11, 14) — a CI pipeline with nothing to gate is just boilerplate.
- Containerization (16) comes after the app is CI-tested — you don't want to Dockerize broken code.
- IaC (17) and K8s (18) come after containers exist — you can't deploy an image that isn't built.
- MLOps (19) is placed after evaluation *and* after K8s, because production experiment tracking and prompt/model registries assume both a metrics pipeline and a place to deploy from.
- Load testing (20) comes after K8s deploy — load testing localhost tells you little about production behavior.

---

## <a name="phase-1"></a>Phase 1 — Environment & Project Scaffolding

**Goal:** A reproducible dev environment and repo skeleton that every later phase writes into.

- [x] Create project repository with the folder structure below
- [x] Set up virtual environment (`poetry` recommended over `conda` for lockfile reproducibility)
- [x] Install core dependencies: `langgraph`, `langchain`, `neo4j`, `pandas`, `fastapi`, `pydantic`
- [x] Set up `.env` for local secrets + `.env.example` committed (never commit real secrets)
- [x] Add `.env`, `data/raw`, `*.db` to `.gitignore`
- [x] Configure `docker-compose.yml` for local services: Neo4j, Redis, Postgres (pgvector), MLflow
- [x] Initialize Git with branch protection on `main` (PR required, at least 1 review or self-review checklist)
- [x] Set up pre-commit hooks: `black`, `ruff`, `isort`, `mypy`
- [x] Initialize DVC for `data/`, pointed at an S3/GCS remote (bucket can be created in Phase 17, stub locally for now)

**Folder structure (final target — built incrementally through the phases below):**

```text
financial-agent/
├── .github/
│   └── workflows/            # Phase 15 — CI/CD
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
├── ingestion/                 # Phase 2
│   ├── parser.py
│   ├── normalizer.py
│   └── graph_loader.py
├── knowledge/                 # Phase 3
│   ├── schema/
│   └── vector_store/
├── agent/
│   ├── tools/                 # Phase 4
│   ├── state/                 # Phase 5 — Pydantic schemas
│   ├── memory/                 # Phase 5
│   ├── graph.py                # Phase 6 — LangGraph StateGraph definition
│   ├── nodes/                  # Phase 6-9 — node functions
│   ├── sandbox/                 # Phase 9
│   ├── mcp/                    # Phase 10
│   └── prompts/                 # versioned prompt files, never inline strings
├── evaluation/                  # Phase 11
├── observability/                # Phase 12
├── api/                          # Phase 14
├── ui/                           # Phase 14
├── infra/                        # Phase 17 — Terraform
│   ├── modules/
│   └── environments/
├── k8s/                          # Phase 18
│   ├── base/
│   └── overlays/
├── docker/                       # Phase 16 — Dockerfiles per service
├── docker-compose.yml
├── docker-compose.prod.yml
├── pyproject.toml
└── README.md
```

---

## <a name="phase-2"></a>Phase 2 — Data Engineering

**Goal:** A clean, versioned, normalized dataset the knowledge graph can be built from.

- [ ] Deep-dive FinQA dataset structure (tables, pre/post text, question, program, gold answer)
- [ ] Write `parser.py` to extract structured records from raw FinQA JSON
- [ ] Write `normalizer.py` to standardize numeric formats, currency, units, fiscal periods
- [ ] Validate parsed output against a hand-checked sample (catch silent parsing errors early)
- [ ] Track dataset versions with DVC (`dvc add data/processed`, commit the `.dvc` file)
- [ ] Document known data quality issues (missing tables, OCR artifacts, ambiguous fiscal years) in `data/README.md`

---

## <a name="phase-3"></a>Phase 3 — Knowledge Graph & Vector Store

**Goal:** Structured (graph) and unstructured (vector) retrieval substrates the agent's tools will query.

- [ ] Design graph schema: entities (Company, FiscalPeriod, LineItem, Metric), relationships (REPORTS, HAS_METRIC, DERIVED_FROM)
- [ ] Set up Neo4j (local via docker-compose for now; AuraDB decision deferred to 3.x below)
- [ ] Write `graph_loader.py` to load normalized data into Neo4j
- [ ] Set up pgvector store for narrative text (pre/post text chunks) with embeddings
- [ ] Validate graph integrity (orphan nodes, missing relationships) and vector recall on a sample query set
- [ ] Document Neo4j AuraDB vs. self-hosted tradeoffs (cost, maintenance, connection limits) as a short ADR (Architecture Decision Record) — this decision feeds Phase 17 IaC

---

## <a name="phase-4"></a>Phase 4 — Agent Tooling

**Goal:** Standalone, individually-tested tools. Nothing here knows about LangGraph yet — tools are plain typed functions.

- [ ] **Graph Retrieval Tool** — natural language → Cypher → structured result
- [ ] **Vector Retrieval Tool** — semantic search over narrative chunks
- [ ] **Context Fusion Tool** — merges graph + vector results into a single context object
- [ ] **Safe Math Tool** — arithmetic/financial calculations without raw `eval()` (this is the precursor to the general sandbox in Phase 9 — keep the interface consistent so Phase 9 can generalize it)
- [ ] **Query Decomposition Tool** — splits multi-hop questions into sub-questions (this output shape is what Phase 7 parallelization will fan out over — design the return type as a list of independent sub-queries now)
- [ ] Unit-test every tool in isolation with known inputs/outputs
- [ ] Add per-tool latency + token/cost logging (structured, e.g. JSON logs) — consumed later by the Phase 12 cost dashboard and Phase 19 MLOps tracking
- [ ] Define each tool's JSON Schema explicitly (input/output) — this is required later for MCP exposure in Phase 10, so do it now rather than retrofitting it

---

## <a name="phase-5"></a>Phase 5 — Agent State Schema & Memory

**Goal:** Define the typed contract every LangGraph node will read/write, and the memory systems that back it. This must exist *before* Phase 6, because LangGraph's `StateGraph` is generic over this exact type.

### 5.1 State Schema

- [ ] Define `AgentState` as a strict Pydantic model — not a loose `TypedDict` of `Any`. Fields: `input`, `scratchpad`, `tool_calls`, `tool_results`, `memory_refs`, `hitl_status`, `trace_id`, `sub_task_results` (for Phase 7)
- [ ] Version the schema (`AgentStateV1`) and write a migration note for how `V2` would extend it
- [ ] Add Pydantic validators so invalid state transitions fail loudly at node boundaries, not silently

### 5.2 Memory Architecture

- [ ] **Short-term / working memory** — current conversation + scratchpad, lives inside `AgentState`, backed by Redis with TTL
- [ ] **Long-term episodic memory** — past sessions, summarized and stored in Postgres, retrievable by recency + relevance
- [ ] **Semantic memory** — facts/entities extracted from interactions; reuse the Phase 3 Neo4j graph rather than standing up a second store
- [ ] **Procedural memory** — which tool sequences worked for which query archetypes; feeds few-shot examples / tool-routing decisions in Phase 6
- [ ] Define explicit promotion policy: what moves from short-term to long-term, and when (e.g. end-of-session summarization job)
- [ ] Expose memory retrieval as a first-class tool (same interface convention as Phase 4 tools, including latency/cost logging and a JSON Schema)

---

## <a name="phase-6"></a>Phase 6 — Orchestration with LangGraph

**Goal:** A single `StateGraph` over `AgentState`, wired to the Phase 4 tools and Phase 5 memory. This is the one and only orchestration mechanism in the system — Phases 7-9 extend it rather than replace it.

- [ ] Define LangGraph nodes: `plan`, `retrieve_graph`, `retrieve_vector`, `fuse_context`, `compute`, `synthesize_answer`, `write_memory`
- [ ] Define conditional edges (e.g. route to `compute` only if the plan requires arithmetic)
- [ ] Wire Phase 4 tools into their corresponding nodes
- [ ] Wire Phase 5 memory read (start of graph) and write (end of graph) as explicit nodes, not side effects buried in other nodes
- [ ] Store every prompt as a versioned file (`agent/prompts/cypher_gen_v1.txt`) — never hardcode prompt strings in node functions
- [ ] Log which prompt version + graph version produced each run (needed for regression debugging in Phase 11 and rollback in Phase 19)
- [ ] Add Redis-backed checkpointing (LangGraph checkpointer) so graph runs are resumable — this checkpointing mechanism is what Phase 8 (HITL) and Phase 9 (sandboxing) will depend on to pause/resume safely
- [ ] Run the full graph end-to-end on 10 hand-picked questions, manually verify traces

---

## <a name="phase-7"></a>Phase 7 — Parallelization

**Goal:** Fan out independent sub-queries as parallel branches inside the *same* graph from Phase 6.

- [ ] Use the Phase 4 Query Decomposition Tool's output (list of sub-queries) to drive a LangGraph fan-out (`Send` API) over parallel `retrieve_graph`/`retrieve_vector` branches
- [ ] Implement a fan-in/aggregation node that merges parallel `sub_task_results` (already a defined field in `AgentState` from Phase 5) before `synthesize_answer`
- [ ] Add a concurrency limit (semaphore or LangGraph's built-in concurrency control) so parallel tool calls don't blow through API rate limits or cost budgets
- [ ] Benchmark one clearly-parallelizable multi-hop query vs. one single-hop query, record the latency delta — this becomes demo material for Phase 21

---

## <a name="phase-8"></a>Phase 8 — Human-in-the-Loop

**Goal:** Add approval gates as interrupts on the Phase 6 graph, using the Phase 6 checkpointer to persist paused state.

- [ ] Define which actions require approval (e.g. destructive graph writes, tool calls above a cost threshold, low-confidence final answers)
- [ ] Use LangGraph's `interrupt_before` / `interrupt_after` on those specific nodes
- [ ] Confirm paused state survives a process restart (this is the actual test of the Phase 6 checkpointer, not just an in-memory pause)
- [ ] Build a minimal approval queue UI (can live in the Phase 14 Streamlit app) — list pending approvals, approve / reject / edit-and-resume
- [ ] Log every HITL event (paused at, resumed at, who approved, decision) — consumed by Phase 12 observability

---

## <a name="phase-9"></a>Phase 9 — Sandboxed Execution

**Goal:** Generalize the Phase 4 Safe Math Tool's isolation guarantee to any dynamically generated code the graph runs (generated Cypher, future code-gen tools).

- [ ] Extract a common `sandbox/` execution interface used by any node that runs generated code
- [ ] Run generated code in an isolated environment — container-per-execution or a hosted sandbox (E2B/Modal) rather than in-process `eval()`
- [ ] Enforce CPU/memory/timeout limits and default-deny network access from inside the sandbox
- [ ] Route the existing Safe Math Tool and the Graph Retrieval Tool's generated Cypher through this same sandbox interface (retrofit, don't leave two isolation mechanisms)
- [ ] Log every sandboxed execution (input, output, exit code, duration) using the same structured logging convention as Phase 4

---

## <a name="phase-10"></a>Phase 10 — MCP Integration

**Goal:** Expose the Phase 4 tools over MCP and consume at least one external MCP server, proving interoperability beyond in-process LangChain tool calls.

- [ ] Wrap the Phase 4 tools (already JSON-Schema-documented) as an MCP server
- [ ] Add an MCP client node/edge in the Phase 6 graph so tool calls can be routed to the local MCP server
- [ ] Consume at least one external MCP server (filesystem, or a public finance MCP) from within the graph, to prove the harness isn't only self-referential
- [ ] Add health checks for the MCP server(s) — reused later in Phase 18 K8s probes

---

## <a name="phase-11"></a>Phase 11 — Evaluation & Testing

**Goal:** Now that the graph runs end-to-end (Phases 6-10), measure it. Evaluation cannot precede a working orchestrator.

- [ ] Build FinQA-style exact-match / execution-accuracy evaluation harness
- [ ] Integrate RAGAS for retrieval quality (context precision/recall) against the Phase 3 retrieval stores
- [ ] Integrate DeepEval or similar for answer correctness/faithfulness
- [ ] Assemble a 50-case regression suite covering single-hop, multi-hop (Phase 7), and HITL-gated (Phase 8) query types
- [ ] Set up MLflow (or Weights & Biases) locally; log every evaluation run: prompt version, graph version, model, retrieval config, metrics
- [ ] Build a leaderboard view comparing runs over time
- [ ] Tag the run that becomes the "production" baseline — this tag is what Phase 19's prompt/model registry references
- [ ] Wire the 50-case regression suite as a script (`evaluation/run_regression.py`) callable from CI (used in Phase 15) that fails if exact-match accuracy drops below a set threshold (e.g. 70%)

---

## <a name="phase-12"></a>Phase 12 — Observability

**Goal:** Full tracing across everything built so far — graph execution, memory, HITL, parallel branches, sandbox, tools.

- [ ] Integrate LangSmith or Arize Phoenix for end-to-end LangGraph trace visualization
- [ ] Structured logging (JSON) across all nodes, tools, and the API layer, correlated by `trace_id` (already a field in `AgentState`)
- [ ] Trace memory reads/writes: which tier was hit, what was retrieved, why (Phase 5)
- [ ] Trace HITL events: paused/resumed timestamps, approver, decision (Phase 8)
- [ ] Trace parallel branch execution as separate spans (Phase 7) — gantt-style view if the tracer supports it
- [ ] Trace sandbox executions: success rate, timeout rate, resource usage (Phase 9)
- [ ] Track token usage and $ cost per query, per tool call, per sandbox execution
- [ ] Set up Prometheus + Grafana; dashboards for latency, error rate, and daily/weekly LLM spend
- [ ] Set a budget alert (Slack webhook if daily spend exceeds threshold)

---

## <a name="phase-13"></a>Phase 13 — Security & Hardening

**Goal:** Harden what's now a fully working, observable system before exposing it via API.

- [ ] Move secrets out of `.env` into a secrets manager (AWS Secrets Manager, GCP Secret Manager, or HashiCorp Vault) — local `.env` remains for pure dev convenience only
- [ ] Add API authentication (API key or OAuth) — applies to the API being built next in Phase 14
- [ ] Add rate limiting (`slowapi`) to the `/query` endpoint
- [ ] Run a dependency vulnerability scan (`pip-audit` or `safety`), wired into CI in Phase 15
- [ ] Confirm the Phase 9 sandbox's network-deny default hasn't regressed (re-test explicitly)

---

## <a name="phase-14"></a>Phase 14 — API & UI

**Goal:** Expose the hardened graph (Phases 6-9, 13) over HTTP and a usable frontend.

- [ ] Build FastAPI backend: `/query` endpoint invokes the Phase 6 graph, `/approvals` endpoint surfaces Phase 8 HITL queue, `/health` for liveness/readiness
- [ ] Build Streamlit frontend: query input + reasoning trace view + the Phase 8 approval queue UI
- [ ] Stream intermediate graph steps to the frontend (LangGraph supports streaming — use it so the UI shows live reasoning, not just a final answer)
- [ ] Manual end-to-end test: submit a query that triggers parallel retrieval (7), an HITL pause (8), and a sandboxed calculation (9), confirm it all surfaces correctly in the UI

---

## <a name="phase-15"></a>Phase 15 — CI/CD

**Goal:** Automate what's currently manual: lint, test, regression-eval, build, deploy. Requires Phase 11 (tests to run), Phase 13 (scans to run), and Phase 14 (something to build) to already exist.

### 15.1 CI Workflow (on every PR)

- [ ] Lint (`ruff`) → type-check (`mypy`) → unit tests (`pytest`) → regression eval (Phase 11's `run_regression.py`, fail under threshold) → dependency vulnerability scan (Phase 13) → build Docker images (dry-run, no push)
- [ ] Add build status badge to README

```yaml
# .github/workflows/ci.yml (sketch)
name: CI
on: [pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: mypy agent/
      - run: pytest tests/ -v
      - run: pip-audit
      - run: python evaluation/run_regression.py --min-accuracy 0.70
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -f docker/api.Dockerfile -t financial-agent-api:ci .
```

### 15.2 CD Workflow (on merge to `main`)

- [ ] Push images to registry (ECR/GCR/Docker Hub) — registry itself is provisioned in Phase 17
- [ ] Deploy to staging (Phase 18 K8s overlay)
- [ ] Run smoke tests against staging
- [ ] Manual approval gate (GitHub Environments protection rule — a nice parallel to the Phase 8 in-app HITL pattern)
- [ ] Deploy to prod

---

## <a name="phase-16"></a>Phase 16 — Containerization

**Goal:** Package the CI-tested app (Phase 15) into images. Comes after CI, not before — never Dockerize code that hasn't passed the pipeline.

- [ ] Dockerfile for FastAPI app (multi-stage: builder + slim runtime)
- [ ] Dockerfile for Streamlit app
- [ ] Dockerfile for the MCP server(s) (Phase 10)
- [ ] Dockerfile for the sandbox execution service (Phase 9) — isolated build, minimal base image
- [ ] `.dockerignore` per service to keep images lean
- [ ] Scan images for vulnerabilities (`trivy` or `docker scout`)
- [ ] Update `docker-compose.yml` with all services (API, UI, MCP, sandbox, Neo4j, Redis, Postgres, MLflow)
- [ ] Test full stack with `docker-compose up`
- [ ] Create `docker-compose.prod.yml` (no dev bind-mounts, real env vars sourced from the Phase 13 secrets manager, resource limits set)

---

## <a name="phase-17"></a>Phase 17 — Infrastructure as Code

**Goal:** Provision cloud infra to run the Phase 16 containers. IaC never precedes containers — there's nothing to deploy otherwise.

- [ ] Terraform modules: VPC/networking, container registry, compute (ECS/EKS or Cloud Run/GKE), Neo4j AuraDB (per the Phase 3 ADR), Redis (ElastiCache/Memorystore), Postgres (RDS/Cloud SQL)
- [ ] Separate `dev` and `prod` Terraform environments
- [ ] Store Terraform state remotely (S3 + DynamoDB lock, or GCS) — also becomes the DVC remote referenced back in Phase 1/2
- [ ] Output registry URL and cluster credentials consumed by Phase 15 CD and Phase 18 K8s config

---

## <a name="phase-18"></a>Phase 18 — Kubernetes Deployment

**Goal:** Deploy the Phase 16 containers onto the Phase 17 infrastructure.

- [ ] Base manifests: `Deployment`, `Service`, `ConfigMap`, `Secret` (via External Secrets Operator pulling from the Phase 13 secrets manager — never plaintext)
- [ ] Separate Deployments for API, UI, MCP server(s), and sandbox service (Phase 9's isolation should extend to pod-level blast-radius isolation, e.g. its own namespace or NetworkPolicy)
- [ ] `HorizontalPodAutoscaler` for the API pod (scale on CPU or request rate)
- [ ] `Ingress` with TLS (cert-manager + Let's Encrypt)
- [ ] Kustomize overlays for `dev`/`staging`/`prod`
- [ ] Liveness/readiness probes hitting each service's `/health` endpoint (defined back in Phase 14 for the API, Phase 10 for MCP)
- [ ] NetworkPolicy denying egress from the sandbox pod by default, consistent with Phase 9's network-deny requirement
- [ ] Optional stretch: Helm chart instead of raw manifests

---

## <a name="phase-19"></a>Phase 19 — MLOps Layer

**Goal:** Turn the local MLflow (Phase 11) and local DVC (Phase 1/2) setup into a centralized, production-facing MLOps layer, now that there's a deployed target (Phase 18) to tie it to.

- [ ] Move MLflow from local to a centralized server (self-hosted on K8s from Phase 18, or a managed service)
- [ ] Point DVC at the cloud remote provisioned in Phase 17 (rather than the local stub from Phase 1)
- [ ] Build a prompt/model registry: tag which prompt version (Phase 6) + model + graph version is "current production" — this is the same tag introduced in Phase 11, now made authoritative and queryable by the running app
- [ ] Automated re-evaluation trigger: if the underlying LLM model version changes, CI (Phase 15) automatically reruns the Phase 11 regression suite and flags drift
- [ ] Wire the production app (Phase 14 API) to read its active prompt/model version from this registry at startup, rather than from a hardcoded config

---

## <a name="phase-20"></a>Phase 20 — Load & Resilience Testing

**Goal:** Test the actual deployed system (Phase 18), not localhost — load testing before a real deploy exists tells you little.

- [ ] Load test the `/query` endpoint with Locust or k6 against the staging K8s deployment; define an acceptable p95 latency under N concurrent users
- [ ] Confirm the Phase 18 HPA scales correctly under the generated load
- [ ] Chaos-test by killing the Redis pod and confirm graceful degradation (the Phase 5 memory read/write nodes should fail closed, not crash the whole graph)
- [ ] Chaos-test by killing the sandbox pod (Phase 9) mid-execution and confirm the graph surfaces a clear error rather than hanging
- [ ] Feed load-test cost data back into the Phase 12 Grafana cost dashboard as a baseline

---

## <a name="phase-21"></a>Phase 21 — Portfolio Packaging

**Goal:** Make sure the engineering work built across Phases 1-20 actually gets seen and understood.

- [ ] Architecture diagram: full data + control flow — FinQA → parser → Neo4j/pgvector → LangGraph (with parallel branches, HITL interrupts, sandbox calls, MCP) → API → UI, plus the infra layer underneath
- [ ] Project README: problem statement, architecture diagram, key design decisions (why knowledge graph + vector hybrid, why LangGraph over a bare loop, why sandboxed execution, why MCP), results table (EM/accuracy vs. FinQA baselines from Phase 11), "how to run locally" section
- [ ] Record a 2-3 minute demo showing: a multi-hop question triggering parallel retrieval, an HITL approval pause, and a sandboxed calculation, with the full reasoning trace visible
- [ ] One blog post on a hard technical decision (e.g. graph/vector conflict resolution, HITL interrupt/resume design, or sandbox isolation tradeoffs)
- [ ] Publish the repo publicly with a clear license and either a pinned live deploy link or an architecture-only public version if hosting cost is a concern

---

## <a name="master-timeline"></a>Master Timeline

```text
Week 1     ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  Phase 1-2
Week 2     ░░░░░░░░░░░░████████████░░░░░░░░░░░░░░░░░░░░  Phase 2-3
Week 3     ░░░░░░░░░░░░░░░░░░░░████████████░░░░░░░░░░░░  Phase 3-4-5
Week 4     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████░░░░  Phase 5-6-7
Week 5     ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░  Phase 7-8-9-10
Week 6     ░░░░░░░░████████████████░░░░░░░░░░░░░░░░░░░░  Phase 10-11-12
Week 7     ░░░░░░░░░░░░░░░░░░░░████████████████░░░░░░░░  Phase 12-13-14-15
Week 8     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████████  Phase 15-16-17-18
Week 9     ████████████████████████████░░░░░░░░░░░░░░░░  Phase 18-19-20
Week 10    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████████  Phase 21
```

*This is a 10-week plan. If timeline is tighter than completeness, use the MVP path below — it front-loads the highest job-signal-per-hour items from every layer (agent internals + platform) rather than fully completing phases in order.*

---

## <a name="mvp-path"></a>MVP → MVP+ → Full Build

### MVP (do these first — a working, demoable agent)

| # | Task | Phase |
|---|---|---|
| 1 | Environment + docker-compose (Neo4j, Redis) | 1 |
| 2 | FinQA parser for 50 samples | 2 |
| 3 | Basic Neo4j graph with core schema | 3 |
| 4 | Graph Retrieval Tool + Safe Math Tool | 4 |
| 5 | `AgentState` schema (minimal fields) | 5 |
| 6 | Basic LangGraph graph (plan → retrieve → compute → answer, no parallel/HITL yet) | 6 |
| 7 | Manual test on 10 questions | 11 (early) |
| 8 | Basic Streamlit UI | 14 |

### MVP+ (do these next — the harness differentiators)

| # | Task | Phase |
|---|---|---|
| 9 | Short-term + episodic memory wired into the graph | 5 |
| 10 | One parallel fan-out/fan-in branch | 7 |
| 11 | One HITL interrupt gate with resume | 8 |
| 12 | Sandbox for the Safe Math Tool (replace in-process eval) | 9 |
| 13 | GitHub Actions CI (lint + test + regression on every PR) | 15 |
| 14 | Dockerfile per service + `.dockerignore` | 16 |
| 15 | One working K8s manifest (tested with `kind`/`minikube` locally) | 18 |

### Full Build

Everything else — vector hybrid retrieval polish, MCP (10), full observability (12), security hardening (13), IaC (17), full K8s (18), MLOps registry (19), load/chaos testing (20), and portfolio packaging (21).

---

## <a name="why-this-matters"></a>Why This Matters for Applied AI Scientist / Agentic AI Roles

These roles sit at the intersection of ML and platform engineering. Interviewers commonly probe: *"How does your agent decide when to ask a human?"*, *"What happens if a tool call fails mid-parallel-branch?"*, *"How do you know a prompt change didn't regress accuracy?"*, *"What stops the agent from executing arbitrary code unsafely?"*, *"How would you actually deploy this?"*

Having real, minimal, working answers — an HITL interrupt that actually pauses and resumes, a sandbox that actually isolates execution, a CI pipeline that actually ran, a K8s manifest that actually deployed to a local cluster, an MLflow run history with a tagged production baseline — is far more convincing than a polished README claiming production-readiness without artifacts behind it. Prioritize *shipping thin, working versions of every layer* (agent internals **and** platform) over gold-plating any single phase.