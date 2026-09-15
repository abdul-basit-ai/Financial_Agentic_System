# Autonomous Financial Reasoning Agent - Benchmark Report

**Dataset Split:** dev

**Total Samples:** 25

**Execution Accuracy (Acc_exe):** 8.00%

**Program Accuracy (Acc_prog):** 8.00%

## Information Retrieval (IR) Multi-Hop Metrics
* **Recall@5:** 17.33%
* **Precision@5:** 20.80%
* **Mean Reciprocal Rank (MRR):** 0.2080
* **NDCG@5:** 0.1728

## Latency & Performance
* **Mean Wall Latency:** 8314.65 ms
* **P95 Latency:** 32797.22 ms

## Causal Failure Taxonomy
| Category | Count | Proportion |
| :--- | :---: | :---: |
| RETRIEVAL_MISS | 22 | 88.0% |
| CORRECT | 2 | 8.0% |
| PLANNING_ERROR | 1 | 4.0% |
