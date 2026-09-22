# Autonomous Financial Reasoning Agent - Benchmark Report

**Dataset Split:** dev

**Total Samples:** 25

**Execution Accuracy (Acc_exe):** 52.00%

**Program Accuracy (Acc_prog):** 40.00%

## Information Retrieval (IR) Multi-Hop Metrics
* **Recall@5:** 84.00%
* **Precision@5:** 56.00%
* **Mean Reciprocal Rank (MRR):** 0.7900
* **NDCG@5:** 0.7516

## Latency & Performance
* **Mean Wall Latency:** 10720.61 ms
* **P95 Latency:** 20780.55 ms

## Causal Failure Taxonomy
| Category | Count | Proportion |
| :--- | :---: | :---: |
| CORRECT | 14 | 56.0% |
| PLANNING_ERROR | 9 | 36.0% |
| RETRIEVAL_MISS | 2 | 8.0% |
