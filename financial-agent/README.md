# financial-agent

Scaffold for a financial reasoning agent with data ingestion, graph loading,
agent orchestration, evaluation, API, and UI modules.

## FinQA Data Pipeline

Generate dataset profile, parsed records, and normalized records from raw FinQA:

```bash
/usr/bin/python3 evaluation/profile_finqa_dataset.py
/usr/bin/python3 ingestion/parser.py
/usr/bin/python3 ingestion/normalizer.py
```

Outputs:

- `evaluation/FINQA_DATASET_PROFILE.md`
- `data/processed/finqa_dataset_profile.json`
- `data/processed/parsed/finqa_*_parsed.jsonl`
- `data/processed/parsed/finqa_parsed_manifest.json`
- `data/processed/normalized/finqa_*_normalized.jsonl`
- `data/processed/normalized/finqa_normalized_summary.json`

## Graph Loading (Neo4j)

Final graph schema and DDL are documented in [docs/GRAPH_SCHEMA.md](docs/GRAPH_SCHEMA.md).

Run a dry-run load summary:

```bash
/usr/bin/python3 ingestion/graph_loader.py --dry-run --limit-per-split 100
```

Run live load:

```bash
/usr/bin/python3 ingestion/graph_loader.py \
	--uri bolt://localhost:7687 \
	--user neo4j \
	--password password \
	--database neo4j
```
