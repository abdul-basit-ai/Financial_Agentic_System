# Worklog

## Goal

Fix the audited defects, implement the agreed Supabase plus Python financial document assistant, improve its UI, and validate the combined result.

## Current instructions

- User explicitly requests committing and pushing all current project changes to the existing GitHub remote. Include project documentation and existing deletions; exclude credentials, runtime data, temporary QA files and the empty Windows nul artifact.
- User explicitly requests subagent research and a comprehensive issue review.
- User now explicitly authorizes fixing all audited issues and improving the UI with subagents. Audit-only restrictions are superseded for repository implementation.
- Use task-appropriate subagent models: Astra for complex reasoning/security, Sol for data/evaluation correctness, Luna for large lower-complexity UI/configuration tasks. Add task-specific application model routing.
- User explicitly requests removal of all code comments. Root owns this mechanical cleanup, including comments in generated Python/SQL; preserve executable behavior and other string/docstring content.
- Target is both a research/portfolio project with rigorous benchmarks and a reliable assistant for uploaded financial reports; retain FinQA as a benchmark.
- User permits hosted model APIs for all uploads, chooses single-user first with a later multi-user path, and prioritizes accuracy while measuring cost and latency before setting limits.
- User permits removing components without demonstrated benchmark benefit, selects English annual reports/SEC filings including scanned pages, and requires strict page/table citations with abstention on ambiguous evidence or units.
- User chooses Supabase Auth, Storage, Postgres/search with a separately hosted Python worker; retain Python pipeline rather than a TypeScript rewrite. User defers arbitrary Python execution and chooses auditable arithmetic over cited evidence.
- Do not read or expose secret values from `.env`; avoid live paid API calls and infrastructure mutations.
- Keep exactly one root worklog. Findings must distinguish reproduced defects, static evidence, and unverified risks.

## Owners

- All implementation peers have completed and handed their files back. Root now owns the integrated tree and final validation.

- Root: assembler; server/auth/durable registries, uploaded-document pipeline, Supabase integration, governance/sandbox/MCP, telemetry, packaging/config/CI and integration validation; sole worklog owner.
- runtime_fix (Astra): legacy LangGraph/runtime/state, nodes except HITL, LLM task routing, memory except checkpoint working.py, safe math/fusion/planning tools and focused runtime tests.
- data_fix (Sol): ingestion, knowledge stores, graph/table/vector client tools, evaluation and focused data/evaluation tests.
- ui_fix (Luna): ui/ and UI/SSE tests; later mechanical deployment/config work only after explicit ownership handoff.
- Runtime peer additionally owns agent/documents/reasoning.py and tests/test_document_reasoning.py for adversarial correctness review. Root owns other document modules.
- UI peer completed its initial UI tests and additionally owns pyproject/config/env/Docker/Compose/CI/K8s/legacy infra/README packaging work. Root generates lockfiles after handback and does not overlap those files.

## Unfinished work

- [ ] Configure and validate hosted Supabase Auth, private Storage and Postgres with a deployed Python worker; no remote infrastructure was changed.
- [ ] Run live candidate benchmarks and ablations against refreshed immutable indexes before selecting production models or claiming SOTA accuracy.
- [ ] Build container images in CI or a Docker-capable host; Docker is unavailable locally.

- [x] Fix runtime grounding, calculation dependencies, planning/recovery, memory/fusion and per-task model routing.
- [x] Repair financial normalization/provenance, database clients and strict reproducible candidate evaluation.
- [x] Enforce authentication/ownership, durable jobs/approvals, audit integrity and bounded execution.
- [x] Implement private report upload, PDF/OCR ingestion, evidence search, cited answers and Supabase adapters.
- [x] Improve UI with authentication, document selection/upload, evidence/citations, results and review recovery.
- [x] Repair sandbox/MCP/health/deployment/configuration and packaged resources; lock dependencies.
- [x] Run full tests, lint/types, artifact checks and complete user-flow verification.

- [x] Review all substantive subsystems across the assigned lanes.
- [x] Run safe local checks and reproduce high impact defects.
- [x] Verify current research against primary sources.
- [x] Ask user for critical product and architecture choices.
- [x] Validate comment removal and scan remaining code/config comments.
- [x] Consolidate prioritized findings with file references and validation limits.

## Decisions

- Prior audit and choices are retained. Functional implementation is now authorized; preserve unrelated dirty user files.
- Peers have non-overlapping write ownership. Root integrates and alone edits worklog. Shared contract changes must be coordinated directly and escalated if unresolved.
- Existing modified tool schemas, AGENTS.md, and paper artifacts are user state; preserve existing changes while applying explicitly authorized comment cleanup and formatting.
- Target architecture: authenticated single-user document assistant using Supabase for identity/files/durable relational evidence and search, plus an independent Python ingestion/LangGraph worker. Worker host/provider remains an implementation selection; no deployment or Supabase mutation is authorized by this audit.
- Start with deterministic arithmetic, canonical evidence references and strict abstention; retain Neo4j/memory/additional agent stages only if ablations demonstrate benefit.

## State

- Push preparation: fetched origin successfully; candidate source credential scan reports no findings. Local credentials/runtime/QA outputs remain ignored. The empty docs/paper/nul artifact is ignored because Windows cannot check out that reserved filename reliably. Committing all project changes and pushing main to the existing origin is authorized.
- Repository inventory and AGENTS.md read. Existing virtual environment and broad test suite found.
- Audit baseline: all three review lanes completed before implementation began.
- Removed 897 Python comments across 74 files and 75 configuration comments across 11 files. Python AST equivalence and line counts verified for every changed Python file. Preserved source strings/docstrings and existing user changes.
- Baseline pip dependency check passes. Baseline Ruff finds two pre-existing issues in untracked docs/paper/check.py.
- All critical product/runtime choices asked in this audit are answered. Exact model and worker hosting provider should be selected through measured quality/cost/latency and deployment fit.
- UI approval-result loss reproduced with Streamlit AppTest and mocked HTTP. Offline wheel build succeeds but omits prompt templates, exemplars, and CSS resources.
- Python comment cleanup formatted with Black at line length 88 and Python 3.11 target, preserving parsed behavior. Also removed two comments in the generated sandbox script, nine embedded SQL comments, and one comment in a code example. Documentation and runtime docstrings otherwise retained.
- Full offline suite after cleanup: 145 passed, 8 skipped. Application/tests Ruff passes, pip check passes, git diff --check passes. In-process formatting and syntax verification passes for 112 Python files with zero Python comment tokens. CLI Black checks stalled and were cancelled; no CLI formatting success claimed.
- After final embedded SQL comment removal, targeted state/memory/integrity checks: 14 passed, 3 skipped; application/tests Ruff and diff whitespace checks pass.
- Mypy reports 17 errors in seven files, including typing assumptions exposed by removed type-ignore comments. Whole-repository Ruff reports two pre-existing issues in untracked docs/paper/check.py.
- Highest-priority verified defects: anonymous thread state/approval access; sandbox host fallback; fabricated arithmetic/semantic grounding; mixed financial units; audit restart index loss; process-local jobs/approvals; inaccurate benchmark scoring and stale-state gates; no actual PDF/upload lifecycle; broken MCP network transport; UI approval result loss; missing packaged prompt/CSS assets.
- Additional findings cover dependent math execution, unvalidated tool payloads, ambiguous cells, ineffective replanning/memory/fusion, provenance offsets and ragged headers, program metric parsing, database transaction/connection handling, approximate retrieval recall controls, incomplete cost/tracing, health/deployment gaps, and missing dependency lockfile.
- Primary sources verified for DocFinQA, FinRank (2026), FinLongDocQA (2026), Docling, structured output, contextual retrieval, and dependency lockfiles.
- Read Supabase skill, current changelog and official documentation through the Supabase documentation tool. Confirmed hosted Edge Functions use TypeScript/Deno; user approved Supabase services plus separate Python worker. No existing Supabase projects inspected or modified.
- CLAUDE.md deletion appeared externally after initial inventory; root did not delete or restore it.
- Audit delivered in previous turn. Implementation now starting from the existing dirty tree; no reset or overwrite of user state.
- Implementation progress: new protected API/auth, owner-scoped durable RecordStore, private file storage, PDF/OCR parser with canonical evidence, durable lease-based Worker, report upload/list/download/delete and queued query routes are written. Integration tests are being adapted to the intentional auth/durability contracts.
- Runtime peer reports 83 runtime/governance checks passing and is reviewing the new document answer engine. Data peer verified scorer semantics against all 883 gold dev execution results, without measuring candidate accuracy. UI peer reports seven focused tests passing.
- Root installed PyMuPDF 1.28.2, checkpoint-sqlite 3.1.3, checkpoint-postgres 3.2.1 and psycopg binary/pool 3.3.9 after approved dependency installation.
- Root replaced unsafe host sandbox execution with explicit denial, removed checkpoint fallback to memory, and implemented transactional append-only audit persistence. Supabase private bootstrap schema is prepared in supabase/schema.sql; not applied remotely.
- No remote deployment or hosted Supabase mutation in this implementation without an explicit applicable authorization; prepare reviewable local schema/setup artifacts and adapters.

- Implementation complete: authenticated owner-scoped API, private report lifecycle, durable transactional jobs/checkpoints/audit, lease fencing/recovery, bounded workers and requests, readiness checks and deterministic cited arithmetic. Unsafe code execution is disabled; MCP is trusted local stdio only; global memory is outside the default graph.
- Document parsing retains page/table/cell/bbox, raw values, scale, quantity kind, period and currency. Native and scanned-table fixtures pass, including real two-pass Tesseract OCR. Conflicting or unsupported OCR layouts abstain; broader scanned-report quality remains unmeasured.
- Strict document reasoning checks model-selected operands and operations against source intent and units and verifies quotations. Runtime supports separate planner/program/verifier/synthesis/document/extraction model settings with nullable unknown costs and opt-in remote tracing.
- FinQA scoring semantics checked against all 8,281 local gold programs (6,251 train, 883 dev, 1,147 test); this validates the interpreter, not candidate agent accuracy. Candidate offline regression executes three mocked-provider graph cases. Frozen manifests and exact case gates prevent stale-state score claims.
- Ledger UI redesigned with private authentication, report upload/status polling/selection, grounded answers, calculation/evidence views, review recovery, persistent saved-analysis discovery and account-change session clearing. Browser upload-to-answer verification recovered the expected operating-income change of 33.3333333333 percent with both cited source cells.
- Local browser QA uses isolated Chrome and synthetic financial statements. A provider connection attempt during early QA failed before producing a response because a dotenv value was loaded; FINAGENT_DISABLE_LLM=1 now prevents initialization and all final QA uses explicit offline configuration. No secret values were printed.
- Supabase private bootstrap SQL, supported Python worker deployment/configuration, nonroot Docker files, archived unsupported AWS scaffolding, updated documentation, CI and package resources are prepared locally. uv.lock resolves 296 packages; the separate locked QA environment installed 146 core/dev packages. Original user virtual environment remains intact.
- Final locked full suite: 251 passed, 3 optional integration checks skipped, 91.11 seconds. Focused UI/security checks: 25 passed, including fresh-session saved-answer recovery and account isolation. Ruff passes; mypy passes all 102 application source files; uv lock --check passes; infrastructure target validator passes; git diff --check passes. Python comment scan found zero comment tokens across 136 source/test/helper files.
- Locked dependency audit reports no known vulnerabilities (local editable project skipped). Wheel build succeeds and runtime prompt/schema/CSS resources were verified in the artifact. Container builds, real Supabase endpoints, database integration and hosted model accuracy/latency/cost were not validated locally.
- Final browser reload exposes persisted research and opens the cited answer without errors. Desktop layout was visually inspected. Attempted narrow viewport emulation did not change the actual viewport, so mobile visual QA remains unverified. Final wheel rebuilt after saved-analysis changes; configuration comment scan also passes. Temporary QA services are stopped after verification.
- User requested a localhost preview. Restarted the local API on 127.0.0.1:8011 and Ledger UI on 127.0.0.1:8511 using the isolated QA data and development token. Hosted model calls remain disabled for this preview; deterministic cited arithmetic is available. Services are left running for user testing.
- Commit hooks exposed a stale-tool mismatch: isolated Black/Ruff/mypy versions differed from uv.lock and the mypy hook lacked runtime dependency types. Replaced hooks with local uv-run checks using locked development dependencies, one Black worker on Windows, Ruff import ordering and a full application mypy run. The first commit was correctly rejected before creating a commit; no check was bypassed.
- Commit validation completed successfully with locked Black, Ruff and full application mypy hooks. All 160 project changes are committed on main and ready for the explicitly requested GitHub push; local secrets, runtime data and temporary artifacts remain excluded.
