# Autonomous Financial Reasoning Agent - Benchmark Report

**Dataset Split:** dev

**Total Samples:** 25

**Execution Accuracy (Acc_exe):** 28.00%

**Program Accuracy (Acc_prog):** 8.00%

## Information Retrieval (IR) Multi-Hop Metrics
* **Recall@5:** 82.00%
* **Precision@5:** 52.00%
* **Mean Reciprocal Rank (MRR):** 0.7300
* **NDCG@5:** 0.7119

## Latency & Performance
* **Mean Wall Latency:** 18921.50 ms
* **P95 Latency:** 25792.09 ms

## Causal Failure Taxonomy
| Category | Count | Proportion |
| :--- | :---: | :---: |
| CORRECT | 7 | 28.0% |
| PLANNING_ERROR | 14 | 56.0% |
| RETRIEVAL_MISS | 4 | 16.0% |
