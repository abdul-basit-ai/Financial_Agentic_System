# Accuracy Plan Smoke Benchmark — 2026-10-20

- Dataset: first 5 records of the FinQA development split
- Execution accuracy: **80.00% (4/5)**
- Program accuracy: 40.00%
- Recall@5: 80.00%
- Mean latency: 16.92 seconds
- P95 latency: 60.20 seconds
- Taxonomy: 4 correct, 1 retrieval miss

The run exercised narrative-aware program generation, train-only exemplars,
grounded candidate validation, verifier selection, and LLM cost telemetry.
`PROGRAM_VERIFIER_MODEL` was not configured, so the verifier used the current
DeepSeek generator model as a fallback. Configure a stronger provider model
before treating this smoke result as the final accuracy target measurement.
