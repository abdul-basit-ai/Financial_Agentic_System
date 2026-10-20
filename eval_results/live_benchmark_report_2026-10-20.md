# Live FinQA Benchmark Report — 2026-10-20

## Scope

- Dataset: first 100 records of the normalized FinQA development split
- Execution: current local source against Docker-hosted Neo4j, pgvector, and Redis
- Important: the already-running API container image was not rebuilt for this run

## Results

| Metric | Live run | Previous recorded run | Change |
| --- | ---: | ---: | ---: |
| Execution accuracy | 51.00% | 52.00% | -1.00 pp |
| Program accuracy | 32.00% | 32.00% | 0.00 pp |
| Recall@5 | 87.92% | 89.42% | -1.50 pp |
| Precision@5 | 56.20% | 57.80% | -1.60 pp |
| MRR | 0.7465 | 0.7645 | -0.0180 |
| NDCG@5 | 0.7463 | 0.7620 | -0.0157 |
| Mean latency | 7,976.63 ms | 11,539.16 ms | -30.87% |
| P95 latency | 18,734.42 ms | 20,605.58 ms | -9.08% |

Median latency was 5,779.76 ms. Observed latency ranged from 2,648.70 ms to
35,952.64 ms.

## Failure attribution

| Category | Count | Share |
| --- | ---: | ---: |
| Correct | 51 | 51% |
| Planning error | 39 | 39% |
| Retrieval miss | 9 | 9% |
| Synthesis hallucination | 1 | 1% |

The largest quality bottleneck is planning/program construction, not retrieval.
Execution accuracy is below the currently configured 70% regression threshold.

## Verification

- Full automated suite: 139 passed, 6 skipped, 1 upstream Starlette/AnyIO deprecation warning
- Live Docker integration suite: 13 passed
- Ruff: clean across application and test sources
- Mypy: clean across 83 source files
- Black: clean for the final evaluator changes
- `pip check`: no broken requirements
- `pip-audit`: no known vulnerabilities (verified during this maintenance pass)
- Docker: API, MLflow, Neo4j, pgvector, and Redis all running

## Corrections made while testing

- Recreated the broken Neo4j container while preserving its data volumes.
- Made pgvector integration checks isolate their own benchmark record.
- Added automatic episodic-memory schema initialization and transaction rollback.
- Reused the cached embedding model instead of loading it for every record.
- Fixed answer parsing so numbers inside explanations cannot be mistaken for the
  final value after `Answer:`.
- Added a regression test for the answer/taxonomy parsing issue.

Cost telemetry recorded `$0.00`, but this should not be treated as a real cost
measurement because the current benchmark result model does not aggregate the
provider usage metadata.
