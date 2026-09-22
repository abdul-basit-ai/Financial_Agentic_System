# Autonomous Financial Reasoning Agent - Benchmark Report

**Dataset Split:** dev

**Total Samples:** 25

**Execution Accuracy (Acc_exe):** 20.00%

**Program Accuracy (Acc_prog):** 8.00%

## Information Retrieval (IR) Multi-Hop Metrics
* **Recall@5:** 84.00%
* **Precision@5:** 58.40%
* **Mean Reciprocal Rank (MRR):** 0.8100
* **NDCG@5:** 0.7733

## Latency & Performance
* **Mean Wall Latency:** 15872.80 ms
* **P95 Latency:** 20731.96 ms

## Causal Failure Taxonomy
| Category | Count | Proportion |
| :--- | :---: | :---: |
| CORRECT | 7 | 28.0% |
| PLANNING_ERROR | 14 | 56.0% |
| RETRIEVAL_MISS | 4 | 16.0% |
