# Financial-Agent Graph Schema

This document finalizes node/relationship design, relationship direction, indexes, constraints, and loader integrity checks.

## Node Types

- Company: `id`, `name`, `ticker`, `sector`
- Report: `id`, `year`, `quarter`, `filing_type`, `source_file`, `record_id`, `question`, `split`
- Table: `id`, `title`, `page_number`
- Row: `id`, `label`, `unit`, `scale`, `category`
- Value: `id`, `amount`, `normalized_amount`, `year`
- TextChunk: `id`, `content`, `page`, `position`, `embedding`
- Metric: `id`, `name`, `category`, `subcategory`

## Relationship Types

- `(Company)-[:FILED]->(Report)`
- `(Report)-[:CONTAINS_TABLE]->(Table)`
- `(Report)-[:HAS_CONTEXT]->(TextChunk)`
- `(Table)-[:HAS_ROW]->(Row)`
- `(Row)-[:HAS_VALUE]->(Value)`
- `(Value)-[:IN_YEAR]->(Report)`
- `(Metric)-[:MEASURED_BY]->(Row)`
- `(Value)-[:NEXT_YEAR]->(Value)`

## Schema DDL (Neo4j)

```cypher
CREATE CONSTRAINT company_id_unique IF NOT EXISTS FOR (c:Company) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT report_id_unique IF NOT EXISTS FOR (r:Report) REQUIRE r.id IS UNIQUE;
CREATE CONSTRAINT table_id_unique IF NOT EXISTS FOR (t:Table) REQUIRE t.id IS UNIQUE;
CREATE CONSTRAINT row_id_unique IF NOT EXISTS FOR (r:Row) REQUIRE r.id IS UNIQUE;
CREATE CONSTRAINT value_id_unique IF NOT EXISTS FOR (v:Value) REQUIRE v.id IS UNIQUE;
CREATE CONSTRAINT textchunk_id_unique IF NOT EXISTS FOR (t:TextChunk) REQUIRE t.id IS UNIQUE;
CREATE CONSTRAINT metric_id_unique IF NOT EXISTS FOR (m:Metric) REQUIRE m.id IS UNIQUE;

CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.name);
CREATE INDEX metric_name IF NOT EXISTS FOR (m:Metric) ON (m.name);
CREATE INDEX value_year IF NOT EXISTS FOR (v:Value) ON (v.year);
```

## Neo4j Configuration

Use these environment variables for loader execution:

- `NEO4J_URI` (default: `bolt://localhost:7687`)
- `NEO4J_USER` (default: `neo4j`)
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE` (default: `neo4j`)

GraphLoader connection tuning flags:

- `--max-connection-pool-size` (default `50`)
- `--connection-timeout` (default `30`)
- `--max-transaction-retry-time` (default `15`)

## Loader Behavior

Implemented in [ingestion/graph_loader.py](../ingestion/graph_loader.py):

- Class: `GraphLoader`
- Batch insert via `UNWIND`
- Idempotent `MERGE` on all node identities
- Relationship mapper across all schema node types
- Automatic `NEXT_YEAR` relationship creation between consecutive `Value` nodes per `Row`
- Integrity checks after loading:
  - orphan Values without Row
  - orphan Rows without Table
  - orphan Tables without Report
  - total counts for each node type and NEXT_YEAR edges

## Run

Dry run:

```bash
/usr/bin/python3 ingestion/graph_loader.py --dry-run --limit-per-split 100
```

Live load:

```bash
/usr/bin/python3 ingestion/graph_loader.py \
  --uri bolt://localhost:7687 \
  --user neo4j \
  --password password \
  --database neo4j
```
