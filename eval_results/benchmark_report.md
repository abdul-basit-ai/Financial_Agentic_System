# Autonomous Financial Reasoning Agent - Benchmark Report

**Dataset Split:** dev

**Total Samples:** 25

**Execution Accuracy (Acc_exe):** 32.00%

**Program Accuracy (Acc_prog):** 8.00%

## Information Retrieval (IR) Multi-Hop Metrics
* **Recall@5:** 84.00%
* **Precision@5:** 57.60%
* **Mean Reciprocal Rank (MRR):** 0.7900
* **NDCG@5:** 0.7611

## Latency & Performance
* **Mean Wall Latency:** 11404.50 ms
* **P95 Latency:** 14018.36 ms

## Causal Failure Taxonomy
| Category | Count | Proportion |
| :--- | :---: | :---: |
| CORRECT | 8 | 32.0% |
| PLANNING_ERROR | 13 | 52.0% |
| RETRIEVAL_MISS | 4 | 16.0% |
