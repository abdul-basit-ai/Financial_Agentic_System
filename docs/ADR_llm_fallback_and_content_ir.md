# ADR — LLM Planner/Synthesizer with Deterministic Fallback & Content-Based IR Metrics

**Status:** Implemented
**Date:** 2026-09-15
**Scope:** Phase 6 (planning), Phase 14 (synthesis), Phase 11 (evaluation)

## Context

Two problems surfaced during the full-project audit:

1. **No LLM in the loop.** The agent was entirely rule-based: the planner was a
   keyword decomposer and the synthesizer a string template, despite prompts,
   `langchain-openai`, and `OPENAI_API_KEY` being provisioned. The goals doc
   describes an LLM-driven planner/synthesizer.

2. **IR metrics structurally zero.** `evaluation/metrics.py` compared retrieved
   identifier strings against gold_inds KEYS (`table_3`, `text_34`). Runtime
   retrieval produces row labels + amounts and chunk text — a different ID
   vocabulary entirely — so recall/precision/MRR/NDCG could never be non-zero
   on real data (existing tests passed only because they used synthetic IDs on
   both sides).

## Decision 1 — LLM planner & synthesizer, fail-open to rules

- New provider module `agent/llm.py`: lazy `ChatOpenAI` from
  `OPENAI_API_KEY` / `OPENAI_MODEL` (default `gpt-4o-mini`) /
  `OPENAI_BASE_URL`. Unavailable ≠ error: no key, construction failure, or
  invocation failure returns `None` and the caller falls back.
- **Planner** (`planner_v2` prompt): LLM emits strict JSON decomposed into
  tool calls; validated by a pydantic model (`LLMPlan`) plus contract checks
  (task_id sequence, no forward dependencies, no bare safe_math without
  retrieval). Any violation → deterministic `decompose_query` fallback.
- **Synthesizer** (`synthesizer_v2` prompt): LLM writes analyst prose over the
  verified evidence bundle. **Anti-hallucination gate:** every numeral in the
  draft must exist in the evidence corpus (question + evidence strings, floats
  compared after comma-stripping). An ungrounded draft is rejected and the
  deterministic template answer ships instead.
- Token counts and estimated cost are logged into the scratchpad for every LLM
  call (Phase 12 hook).

**Why fail-open:** evaluation and CI must run offline (no API key, no network).
The deterministic paths are the same code paths that were correct before; the
LLM is strictly an enhancement layer on top.

## Decision 2 — Content-overlap IR grading

- New `compute_ir_metrics_content(retrieved_texts, gold_texts, k)`: a gold
  evidence string is *covered* by a retrieved text when they share at least one
  significant number (if the gold carries numbers) AND one content word.
- Recall = gold items covered by top-k; precision = top-k slots that cover any
  gold; MRR = first covering slot; **NDCG credits each gold item at most once**
  (duplicate retrievals of the same gold item would otherwise push DCG above
  IDCG — a bug the tests caught on first run).
- The old ID-space `compute_ir_metrics` is kept for callers that genuinely
  share one ID vocabulary; `EvaluationRecord` gained `gold_evidence` (the
  gold_indices VALUES) alongside `gold_inds` (keys, kept for diagnostics).

## Consequences

- With `OPENAI_API_KEY` unset, behavior is byte-identical to the prior
  deterministic agent; with it set, plans/synthesis come from the LLM under
  strict validation.
- IR metrics on real FinQA data are now meaningful and can gate regressions.
- The regression fixture chain (runner `--save-states` → `run_regression.py`
  replay → `gate.py` statistical comparison) is now actually wired end-to-end.

## Alternatives considered

- **LLM-only planner without fallback** — rejected: makes eval/CI depend on a
  paid API and network.
- **Embedding-similarity evidence matching** — rejected for now: token-overlap
  is deterministic, explainable, and free; embeddings can replace
  `evidence_hit` later behind the same function signature.
- **Fixing ID spaces at the data layer** (writing FinQA key-compatible IDs into
  the graph) — rejected: the runtime ID vocabulary (labels + amounts) is more
  useful to the agent than FinQA's positional keys; grading should adapt, not
  the data.
