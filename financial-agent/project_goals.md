# Autonomous Financial Reasoning Agent — Project Phases & Tasks 

> **What changed from v1:** Added CI/CD, IaC, container registry, Kubernetes, MLOps (experiment/data/prompt tracking), secrets management, load testing, and portfolio-presentation deliverables. Changes are integrated into existing phases where they belong, plus a new **Phase 8** for cloud-native deployment. Original items are unchanged; new items are marked **🆕**.

---

## Overview of Phases

| Phase | Name | Duration | Priority |
| --- | --- | --- | --- |
| Phase 1 | Foundation & Data Engineering | Week 1-2 | Critical |
| Phase 2 | Knowledge Graph Construction | Week 2-3 | Critical |
| Phase 3 | Agent Tooling | Week 3-4 | Critical |
| Phase 4 | Agent Orchestration & ReAct Loop | Week 4-5 | Critical |
| Phase 5 | Evaluation & Testing | Week 5-6 | Important |
| Phase 6 | Observability & Production Hardening | Week 6-7 | Important |
| Phase 7 | UI & Deployment | Week 7-8 | Final |
| Phase 8 🆕 | MLOps & Cloud-Native Deployment | Week 8-9 | Important |
| Phase 9 🆕 | Portfolio Packaging | Week 9 | Final |

---

## Phase 1 — Foundation & Data Engineering

**Goal:** Understand the dataset deeply and build a clean, normalized data pipeline.

### 1.1 Environment Setup

- [ ] Create project repository with proper folder structure
- [ ] Set up virtual environment (`poetry` or `conda`)
- [ ] Install core dependencies (`langchain`, `neo4j`, `pandas`, `fastapi`)
- [ ] Set up `.env` file for API keys and credentials
- [ ] 🆕 Set up `.env.example` (never commit real secrets) and add `.env` to `.gitignore`
- [ ] Configure Docker Compose for Neo4j + Redis + pgvector
- [ ] 🆕 Initialize Git repo with branch protection rules (`main` protected, PR required)
- [ ] 🆕 Set up pre-commit hooks (`black`, `ruff`, `isort`, `mypy`)
- [ ] 🆕 Initialize DVC (Data Version Control) for `data/` directory, pointed at S3/GCS remote

**Folder Structure (updated):**

```text
financial-agent/
├── .github/
│   └── workflows/         # 🆕 CI/CD pipelines
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
├── ingestion/
│   ├── parser.py
│   ├── normalizer.py
│   └── graph_loader.py
├── agent/
│   ├── tools/
│   ├── orchestrator.py
│   └── prompts/           # 🆕 versioned prompt files, not inline strings
├── evaluation/
├── api/
├── ui/
├── infra/                 # 🆕 Terraform / IaC
│   ├── modules/
│   └── environments/
├── k8s/                   # 🆕 Kubernetes manifests
│   ├── base/
│   └── overlays/
├── docker/                 # 🆕 Dockerfiles per service
├── docker-compose.yml
└── docker-compose.prod.yml # 🆕
```

### 1.2 – 1.4

*(unchanged from v1 — FinQA analysis, parsing pipeline, validation)*

- [ ] 🆕 Track dataset versions with DVC (`dvc add data/processed`, commit `.dvc` file)

---

## Phase 2 — Knowledge Graph Construction

*(unchanged — schema design, Neo4j setup, graph loader, vector store, validation)*

- [ ] 🆕 Document Neo4j AuraDB vs self-hosted tradeoffs (cost, maintenance, connection limits) as a short ADR (Architecture Decision Record)

---

## Phase 3 — Agent Tooling

*(unchanged — Graph Retrieval, Vector Retrieval, Context Fusion, Safe Math, Query Decomposition, Tool Testing)*

- [ ] 🆕 Add per-tool latency + token cost logging (used later by Phase 8 cost dashboard)

---

## Phase 4 — Agent Orchestration & ReAct Loop

*(unchanged core — AgentState, LangGraph nodes, prompt engineering, ReAct testing, Redis caching)*

### 4.3 Prompt Engineering — additions

- [ ] 🆕 Store every prompt as a versioned file (`prompts/cypher_gen_v1.txt`) — never hardcode in Python
- [ ] 🆕 Log which prompt version produced each answer (needed for regression debugging)

---

## Phase 5 — Evaluation & Testing

*(unchanged core — FinQA eval, RAGAS, DeepEval, metrics, evaluation runs, regression suite)*

### 5.5 🆕 Experiment Tracking

- [ ] Set up MLflow or Weights & Biases
- [ ] Log every evaluation run: prompt version, model, retrieval config, metrics
- [ ] Build a leaderboard view comparing runs over time
- [ ] Tag the run that becomes the "production" baseline

### 5.6 🆕 CI-Gated Regression Testing

- [ ] Wire the 50-case regression suite into a GitHub Actions job
- [ ] Fail the PR build if exact-match accuracy drops below threshold (e.g. 70%)

---

## Phase 6 — Observability & Production Hardening

*(unchanged core — LangSmith, Arize Phoenix, structured logging, Prometheus, Grafana)*

### 6.3 🆕 Cost & Token Observability

- [ ] Track token usage and $ cost per query, per tool call
- [ ] Add a Grafana panel for daily/weekly LLM spend
- [ ] Set a budget alert (e.g. Slack webhook if daily spend exceeds threshold)

### 6.4 🆕 Security Hardening

- [ ] Move secrets out of `.env` into a secrets manager (AWS Secrets Manager, GCP Secret Manager, or HashiCorp Vault)
- [ ] Add API authentication (API key or OAuth) to the FastAPI backend
- [ ] Add basic rate limiting (e.g. `slowapi`) to `/query`
- [ ] Run a dependency vulnerability scan (`pip-audit` or `safety`) in CI

---

## Phase 7 — UI & Deployment

### 7.1 – 7.2

*(unchanged — FastAPI backend, Streamlit frontend)*

### 7.3 Dockerization — expanded

- [ ] Write Dockerfile for FastAPI app (multi-stage build: builder + slim runtime)
- [ ] Write Dockerfile for Streamlit app
- [ ] 🆕 Add `.dockerignore` to keep images lean
- [ ] 🆕 Scan images for vulnerabilities (`docker scout` or `trivy`)
- [ ] Update docker-compose.yml with all services
- [ ] Test full stack with `docker-compose up`
- [ ] 🆕 Create a separate `docker-compose.prod.yml` (no dev bind-mounts, real env vars, resource limits)

### 7.4 Deployment — expanded

- [ ] Deploy to AWS EC2 or GCP Cloud Run *(kept as a fast-path option)*
- [ ] Set up Neo4j AuraDB (managed cloud)
- [ ] Configure environment variables in cloud
- [ ] Write README.md with setup instructions
- [ ] 🆕 CI/CD pipeline (GitHub Actions) — see Phase 8 for full spec, this task now points there

---

## Phase 8 🆕 — MLOps & Cloud-Native Deployment

**Goal:** Turn the working prototype into something that looks and runs like a production system — this is the phase that most differentiates a portfolio project aimed at Applied AI Scientist / Agentic AI roles.

### 8.1 Infrastructure as Code

- [ ] Write Terraform modules for: VPC/networking, container registry, compute (ECS/EKS or Cloud Run/GKE), Neo4j AuraDB provisioning, Redis (ElastiCache/Memorystore)
- [ ] Separate `dev` and `prod` Terraform environments
- [ ] Store Terraform state remotely (S3 + DynamoDB lock, or GCS)

### 8.2 CI/CD Pipeline (GitHub Actions)

- [ ] **CI workflow** on every PR: lint → type-check → unit tests → regression eval (from 5.6) → build Docker images
- [ ] **CD workflow** on merge to `main`: push images to registry (ECR/GCR/Docker Hub) → deploy to staging → run smoke tests → manual approval gate → deploy to prod
- [ ] Add build status badges to README

```yaml
# .github/workflows/ci.yml (sketch)
name: CI
on: [pull_request]
jobs:
  test:
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: pytest tests/ -v
      - run: python evaluation/run_regression.py --min-accuracy 0.70
  build:
    needs: test
    steps:
      - run: docker build -t financial-agent-api .
```

### 8.3 Kubernetes

- [ ] Write base manifests: `Deployment`, `Service`, `ConfigMap`, `Secret` (via External Secrets Operator, not plaintext)
- [ ] Add `HorizontalPodAutoscaler` for the API pod (scale on CPU or request rate)
- [ ] Add `Ingress` with TLS (cert-manager + Let's Encrypt)
- [ ] Use Kustomize overlays for `dev`/`staging`/`prod` environment differences
- [ ] Add liveness/readiness probes hitting the `/health` endpoint
- [ ] 🆕 optional stretch: Helm chart instead of raw manifests, if you want the extra portfolio signal

### 8.4 MLOps Layer

- [ ] Centralize experiment tracking (MLflow server, self-hosted or managed)
- [ ] Data versioning with DVC connected to cloud remote storage
- [ ] Prompt/model registry: tag which prompt + model combo is "current production"
- [ ] Automated re-evaluation trigger: if the underlying LLM model version changes, CI reruns the regression suite automatically

### 8.5 Load & Resilience Testing

- [ ] Load test the `/query` endpoint with Locust or k6 (target: define acceptable p95 latency under N concurrent users)
- [ ] Chaos-test one failure mode (e.g. kill the Redis pod) and confirm graceful degradation from Phase 6

---

## Phase 9 🆕 — Portfolio Packaging

**Goal:** Make sure the engineering work actually gets seen and understood by whoever is evaluating you for a role.

- [ ] Draw an architecture diagram (data flow: FinQA → parser → Neo4j/pgvector → agent → API → UI, plus infra layer)
- [ ] Write a project README with: problem statement, architecture diagram, key design decisions (why knowledge graph + vector hybrid, why safe-math sandboxing), results table (EM/accuracy vs FinQA baselines), and a "how to run locally" section
- [ ] Record a 2-3 minute demo video/GIF showing a multi-hop question being answered with full reasoning trace
- [ ] Write one blog post walking through a hard technical decision (e.g. graph/vector conflict resolution, or Cypher-generation failure handling) — this is often what gets read before the code
- [ ] Publish the repo publicly with a clear license and pinned deploy demo link (if kept live) or an architecture-only public version if cost is a concern

---

## Master Task Timeline (updated)

```text
Week 1    ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  Phase 1
Week 2    ████████████████░░░░░░░░░░░░░░░░░░░░  Phase 1 + Phase 2 Start
Week 3    ░░░░░░░░████████████████░░░░░░░░░░░░  Phase 2 + Phase 3 Start
Week 4    ░░░░░░░░░░░░░░░░████████████████░░░░  Phase 3 + Phase 4 Start
Week 5    ████████████████████████░░░░░░░░░░░░  Phase 4 + Phase 5
Week 6    ░░░░░░░░░░░░░░░░████████████████░░░░  Phase 5 + Phase 6
Week 7    ░░░░░░░░░░░░░░░░░░░░░░░░████████░░░░  Phase 6 + Phase 7
Week 8    ████████████████░░░░░░░░░░░░░░░░░░░░  Phase 7 + Phase 8 Start
Week 9    ░░░░░░░░░░░░░░░░████████████████████  Phase 8 + Phase 9
```

*Note: this extends the original 8-week plan to 9. If timeline matters more than completeness, treat Phase 8/9 as optional stretch — Phase 8.2 (CI/CD) and 8.3 (basic K8s manifests, even unused) give the most job-signal per hour invested.*

---

## Minimum Viable Agent (MVP) — Do These First

*(unchanged from v1)*

| Priority | Task |
| --- | --- |
| 1 | Environment setup + Docker |
| 2 | FinQA parser for 50 samples |
| 3 | Basic Neo4j graph with core schema |
| 4 | Graph Retrieval Tool (basic Cypher) |
| 5 | Safe Math Tool |
| 6 | Simple ReAct loop (no LangGraph yet) |
| 7 | Test on 10 questions manually |
| 8 | Basic Streamlit UI |

**🆕 MVP+ (do these next, before deep-diving evaluation/observability):**

| Priority | Task |
| --- | --- |
| 9 | GitHub Actions CI (lint + test on every PR) |
| 10 | Dockerfile per service + `.dockerignore` |
| 11 | One working k8s manifest (even if only tested with `kind`/`minikube` locally) |

Then iterate to add hybrid retrieval, LangGraph, full evaluation, MLOps tracking, and cloud deployment on top of the MVP+.

---

## Why This Matters for Applied AI Scientist / Agentic AI Roles

These roles increasingly sit at the intersection of ML and platform engineering. Interviewers commonly probe: *"How would you deploy this?"*, *"How do you know if a prompt change regressed something?"*, *"What happens if this tool call fails in prod?"* Having real (even minimal) answers — a CI pipeline that actually ran, a k8s manifest that actually deployed to a local cluster, an MLflow run history — is far more convincing than a polished README claiming production-readiness without artifacts to back it up. Prioritize *shipping thin versions* of Phase 8 items over gold-plating Phases 1-7 further.