# Autonomous Financial Reasoning Agent - Benchmark Report

**Dataset Split:** dev

**Total Samples:** 100

**Execution Accuracy (Acc_exe):** 52.00%

**Program Accuracy (Acc_prog):** 32.00%

## Information Retrieval (IR) Multi-Hop Metrics
* **Recall@5:** 89.42%
* **Precision@5:** 57.80%
* **Mean Reciprocal Rank (MRR):** 0.7645
* **NDCG@5:** 0.7620

## Latency & Performance
* **Mean Wall Latency:** 11539.16 ms
* **P95 Latency:** 20605.58 ms

## Causal Failure Taxonomy
| Category | Count | Proportion |
| :--- | :---: | :---: |
| CORRECT | 53 | 53.0% |
| PLANNING_ERROR | 38 | 38.0% |
| RETRIEVAL_MISS | 8 | 8.0% |
| SYNTHESIS_HALLUCINATION | 1 | 1.0% |
