# Data Documentation — FinQA Dataset

## Overview

The Financial Agent consumes the FinQA dataset (financial question-answering over
SEC 10-K/10-Q excerpts). Raw data lives in `data/raw/FinQA-main/` (committed);
processed artifacts live in `data/processed/` (git-ignored, rebuilt via
`python -m ingestion.parser` then `python -m ingestion.normalizer`).

## Splits

| Split | Records | Answers | Programs | Notes |
|---|---|---|---|---|
| train | 6251 | 6203 | 6251 | 48 records lack `qa.answer` (unanswerable) |
| dev | 883 | 871 | 883 | primary evaluation split |
| test | 1147 | 1133 | 1147 | held-out |
| private_test | 919 | 0 | 0 | hidden leaderboard split — no gold answers by design |

Record counts are verified by `tests/test_data_pipeline.py` (zero-drop audit).

## Record Schema (normalized)

Produced by `ingestion/normalizer.py` — one JSON object per line:

- `record_id` — stable FinQA ID (e.g. `AMZN/2020/page_12.pdf-1`)
- `document` — question, answer_text, answer_numeric(+base), execution_answer,
  question_token_count, question_types
- `reasoning` — program (FinQA DSL), program_re, program_ops, **gold_indices**
  (dict: evidence key -> evidence text; keys are the gold evidence IDs)
- `context` — pre_text / post_text sentence lists, chunked context
- `table` — normalized rows: each cell carries `raw`, `numeric_value`,
  `numeric_value_base`, `unit_label`, `scale_multiplier`, `is_percentage`,
  `is_missing`
- `entities` — company_names, fiscal_years, metric_names(+resolved), units
- `quality` — edge_case_tags (see below)

## Known Data Quality Issues

Tracked per-record in `quality.edge_case_tags` and aggregated in
`data/processed/normalized/finqa_normalized_summary.json`:

1. **Duplicated percent format** — FinQA restates percentages as
   `"11.1% ( 11.1 % )"`. The plain numeric regex silently dropped these;
   `FIN_DUP_PCT_RE` in the normalizer now parses them. (Fixed — see git log
   "Fix silent percent-scaling".)
2. **Percent/unit-scale interaction** — table context like "in millions" was
   multiplying percentage cells (`21%` → base 21,000,000). Percentages are now
   never scaled (`numeric_value_base == numeric_value` for percents).
   (Fixed.)
3. **Missing value cells** (~64% of records) — cells containing `-`, `--`,
   `NA`, `NM`, `$ -` etc. Tagged `missing_value_cells`; kept as
   `is_missing=True` rather than dropped so the agent can reason over gaps.
4. **Multi-year tables** (~57%) — fiscal-year columns detected via header year
   extraction; enables `NEXT_YEAR` graph links.
5. **Possible merged cells** (~27%) — empty first cell with populated rest of
   row; tagged for parser awareness.
6. **Missing programs** — all 919 private_test records have no program/answer
   (by design, not corruption).
7. **Unanswerable questions** — 48 train / 12 dev / 14 test records have empty
   `qa.answer`; `execution_answer` remains present.
8. **Mixed magnitude scales** — single tables mix thousands/millions/billions
   rows. Downstream implication: risk-engine outlier detection clusters by
   order of magnitude before MAD (see `agent/guardrails/risk_engine.py`).
9. **OCR / tokenization artifacts** — text is pre-tokenized with spurious
   spaces (e.g. "1 , 000"); `clean_text` and the numeric regex tolerate this;
   narrative chunks inherit the spacing as-is.
10. **Ambiguous fiscal years** — some tables contain year ranges or
    non-calendar fiscal labels in headers; `parse_year` extracts 19xx/20xx
    only, and unmatched columns yield `year=None` values (visible in graph
    Value nodes).

## Numeric Semantics

- `numeric_value` — parsed value in table-native units (auditor-comparable).
- `numeric_value_base` — value scaled to absolute units using the detected
  table context (millions → ×1e6). Percentages exempt (see #2).
- Execution-answer parity with FinQA `exe_ans` is preserved: values are never
  divided by 100 during parsing; percentage scale equivalence is handled in
  evaluation (`evaluation/metrics.py`).

## Rebuilding

```bash
python -m ingestion.parser      # raw JSON -> data/processed/parsed/*.jsonl
python -m ingestion.normalizer  # parsed -> data/processed/normalized/*.jsonl
python tests/test_data_pipeline.py --splits train dev test  # audit CLI
```

DVC tracks `data/` via `data.dvc` (remote is a Phase 17 item).
