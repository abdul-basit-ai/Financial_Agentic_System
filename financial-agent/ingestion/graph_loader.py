"""Neo4j GraphLoader for financial-agent.

Implements the phase-2 schema with idempotent batch loading and integrity checks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

SPLITS = ["train", "dev", "test", "private_test"]
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			yield json.loads(line)


def slugify(value: str) -> str:
	v = (value or "").strip().lower()
	v = re.sub(r"[^a-z0-9]+", "_", v)
	v = re.sub(r"_+", "_", v).strip("_")
	return v or "unknown"


def safe_list(value: Any) -> List[Any]:
	return value if isinstance(value, list) else []


def parse_year(value: Any) -> Optional[int]:
	if value is None:
		return None
	if isinstance(value, int):
		if 1900 <= value <= 2100:
			return value
		return None
	m = YEAR_RE.search(str(value))
	if not m:
		return None
	try:
		return int(m.group(0))
	except ValueError:
		return None


def infer_row_category(label: str) -> str:
	t = (label or "").lower()
	if any(k in t for k in ["revenue", "sales"]):
		return "revenue"
	if any(k in t for k in ["income", "earnings", "profit"]):
		return "profitability"
	if any(k in t for k in ["asset", "liabil", "equity", "debt"]):
		return "balance_sheet"
	if any(k in t for k in ["cash flow", "cash"]):
		return "cash_flow"
	if any(k in t for k in ["tax", "interest", "expense", "cost"]):
		return "cost_expense"
	return "other"


@dataclass
class GraphLoadStats:
	reports: int = 0
	tables: int = 0
	rows: int = 0
	values: int = 0
	chunks: int = 0
	metrics: int = 0
	companies: int = 0
	next_year_links: int = 0


def build_graph_payload(rec: Dict[str, Any]) -> Dict[str, Any]:
	doc = rec.get("document", {}) if isinstance(rec.get("document"), dict) else {}
	ctx = rec.get("context", {}) if isinstance(rec.get("context"), dict) else {}
	table = rec.get("table", {}) if isinstance(rec.get("table"), dict) else {}
	entities = rec.get("entities", {}) if isinstance(rec.get("entities"), dict) else {}

	record_id = str(rec.get("record_id", ""))
	filename = str(doc.get("filename", ""))
	question = str(doc.get("question", ""))
	fiscal_years = safe_list(entities.get("fiscal_years"))
	report_year = parse_year(fiscal_years[0]) if fiscal_years else None

	company_names = safe_list(entities.get("company_names"))
	company_name = company_names[0] if company_names else f"Unknown Company ({filename or record_id})"
	company_id = f"company::{slugify(company_name)}"

	report_id = f"report::{slugify(filename or record_id)}"
	table_id = f"table::{record_id}"

	table_rows = safe_list(table.get("rows"))
	header = safe_list(table.get("header"))
	metric_names = safe_list(entities.get("metric_names_resolved") or entities.get("metric_names"))

	chunks: List[Dict[str, Any]] = []
	for idx, ch in enumerate(safe_list(ctx.get("chunks"))):
		if not isinstance(ch, dict):
			continue
		chunks.append(
			{
				"id": f"chunk::{record_id}::{idx}",
				"content": str(ch.get("text", "")),
				"page": None,
				"position": idx,
				"embedding": None,
			}
		)

	rows: List[Dict[str, Any]] = []
	values: List[Dict[str, Any]] = []
	metric_row_links: List[Dict[str, Any]] = []

	for ridx, row in enumerate(table_rows):
		if not isinstance(row, list) or len(row) == 0:
			continue
		row_label = str(row[0].get("raw", "")).strip() if isinstance(row[0], dict) else ""
		if not row_label:
			row_label = f"row_{ridx}"

		row_id = f"row::{record_id}::{ridx}"
		rows.append(
			{
				"id": row_id,
				"label": row_label,
				"unit": "base",
				"scale": "raw",
				"category": infer_row_category(row_label),
			}
		)

		for metric in metric_names:
			m = str(metric)
			if m and (m.replace("_", " ") in row_label.lower() or row_label.lower() in m.replace("_", " ")):
				metric_row_links.append({"metric_id": f"metric::{slugify(m)}", "row_id": row_id})

		for cidx, cell in enumerate(row[1:], start=1):
			if not isinstance(cell, dict):
				continue
			amount = cell.get("numeric_value")
			normalized_amount = cell.get("numeric_value_base")
			if amount is None and normalized_amount is None:
				continue

			year = parse_year(header[cidx] if cidx < len(header) else None)
			value_id = f"value::{record_id}::{ridx}::{cidx}"
			values.append(
				{
					"id": value_id,
					"row_id": row_id,
					"amount": amount,
					"normalized_amount": normalized_amount,
					"year": year,
				}
			)

	# Fallback metric links if no lexical match found.
	if not metric_row_links and metric_names and rows:
		for m in metric_names:
			metric_row_links.append({"metric_id": f"metric::{slugify(str(m))}", "row_id": rows[0]["id"]})

	metrics = [
		{
			"id": f"metric::{slugify(str(m))}",
			"name": str(m),
			"category": infer_row_category(str(m)),
			"subcategory": "derived",
		}
		for m in metric_names
	]

	return {
		"company": {
			"id": company_id,
			"name": company_name,
			"ticker": None,
			"sector": None,
		},
		"report": {
			"id": report_id,
			"year": report_year,
			"quarter": None,
			"filing_type": "annual_report",
			"source_file": filename,
			"record_id": record_id,
			"question": question,
			"split": str(rec.get("split", "")),
		},
		"table": {
			"id": table_id,
			"title": question[:120] if question else "Financial Table",
			"page_number": None,
		},
		"chunks": chunks,
		"rows": rows,
		"values": values,
		"metrics": metrics,
		"metric_row_links": metric_row_links,
	}


class GraphLoader:
	def __init__(
		self,
		uri: str,
		user: str,
		password: str,
		database: str = "neo4j",
		max_connection_pool_size: int = 50,
		connection_timeout: float = 30.0,
		max_transaction_retry_time: float = 15.0,
	) -> None:
		self.database = database
		self._driver = None
		self._uri = uri
		self._user = user
		self._password = password
		self._max_connection_pool_size = max_connection_pool_size
		self._connection_timeout = connection_timeout
		self._max_transaction_retry_time = max_transaction_retry_time

	def connect(self) -> None:
		from neo4j import GraphDatabase

		self._driver = GraphDatabase.driver(
			self._uri,
			auth=(self._user, self._password),
			max_connection_pool_size=self._max_connection_pool_size,
			connection_timeout=self._connection_timeout,
			max_transaction_retry_time=self._max_transaction_retry_time,
		)

	def close(self) -> None:
		if self._driver is not None:
			self._driver.close()

	def create_schema(self) -> None:
		assert self._driver is not None
		queries = [
			"CREATE CONSTRAINT company_id_unique IF NOT EXISTS FOR (c:Company) REQUIRE c.id IS UNIQUE",
			"CREATE CONSTRAINT report_id_unique IF NOT EXISTS FOR (r:Report) REQUIRE r.id IS UNIQUE",
			"CREATE CONSTRAINT table_id_unique IF NOT EXISTS FOR (t:Table) REQUIRE t.id IS UNIQUE",
			"CREATE CONSTRAINT row_id_unique IF NOT EXISTS FOR (r:Row) REQUIRE r.id IS UNIQUE",
			"CREATE CONSTRAINT value_id_unique IF NOT EXISTS FOR (v:Value) REQUIRE v.id IS UNIQUE",
			"CREATE CONSTRAINT textchunk_id_unique IF NOT EXISTS FOR (t:TextChunk) REQUIRE t.id IS UNIQUE",
			"CREATE CONSTRAINT metric_id_unique IF NOT EXISTS FOR (m:Metric) REQUIRE m.id IS UNIQUE",
			"CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.name)",
			"CREATE INDEX metric_name IF NOT EXISTS FOR (m:Metric) ON (m.name)",
			"CREATE INDEX value_year IF NOT EXISTS FOR (v:Value) ON (v.year)",
		]
		with self._driver.session(database=self.database) as session:
			for q in queries:
				session.run(q)

	@staticmethod
	def _merge_batch_tx(tx: Any, payloads: List[Dict[str, Any]]) -> None:
		tx.run(
			"""
			UNWIND $rows AS row
			MERGE (c:Company {id: row.id})
			SET c.name = row.name,
				c.ticker = row.ticker,
				c.sector = row.sector
			""",
			rows=[p["company"] for p in payloads],
		)

		tx.run(
			"""
			UNWIND $rows AS row
			MERGE (r:Report {id: row.id})
			SET r.year = row.year,
				r.quarter = row.quarter,
				r.filing_type = row.filing_type,
				r.source_file = row.source_file,
				r.record_id = row.record_id,
				r.question = row.question,
				r.split = row.split
			""",
			rows=[p["report"] for p in payloads],
		)

		tx.run(
			"""
			UNWIND $rows AS row
			MATCH (c:Company {id: row.company_id})
			MATCH (r:Report {id: row.report_id})
			MERGE (c)-[:FILED]->(r)
			""",
			rows=[{"company_id": p["company"]["id"], "report_id": p["report"]["id"]} for p in payloads],
		)

		tx.run(
			"""
			UNWIND $rows AS row
			MERGE (t:Table {id: row.id})
			SET t.title = row.title,
				t.page_number = row.page_number
			""",
			rows=[p["table"] for p in payloads],
		)

		tx.run(
			"""
			UNWIND $rows AS row
			MATCH (r:Report {id: row.report_id})
			MATCH (t:Table {id: row.table_id})
			MERGE (r)-[:CONTAINS_TABLE]->(t)
			""",
			rows=[{"report_id": p["report"]["id"], "table_id": p["table"]["id"]} for p in payloads],
		)

		chunk_rows: List[Dict[str, Any]] = []
		for p in payloads:
			for c in p["chunks"]:
				chunk_rows.append({"report_id": p["report"]["id"], **c})

		if chunk_rows:
			tx.run(
				"""
				UNWIND $rows AS row
				MERGE (tc:TextChunk {id: row.id})
				SET tc.content = row.content,
					tc.page = row.page,
					tc.position = row.position,
					tc.embedding = row.embedding
				WITH row, tc
				MATCH (r:Report {id: row.report_id})
				MERGE (r)-[:HAS_CONTEXT]->(tc)
				""",
				rows=chunk_rows,
			)

		row_rows: List[Dict[str, Any]] = []
		for p in payloads:
			for r in p["rows"]:
				row_rows.append({"table_id": p["table"]["id"], **r})

		if row_rows:
			tx.run(
				"""
				UNWIND $rows AS row
				MERGE (rw:Row {id: row.id})
				SET rw.label = row.label,
					rw.unit = row.unit,
					rw.scale = row.scale,
					rw.category = row.category
				WITH row, rw
				MATCH (t:Table {id: row.table_id})
				MERGE (t)-[:HAS_ROW]->(rw)
				""",
				rows=row_rows,
			)

		value_rows: List[Dict[str, Any]] = []
		for p in payloads:
			for v in p["values"]:
				value_rows.append({"report_id": p["report"]["id"], **v})

		if value_rows:
			tx.run(
				"""
				UNWIND $rows AS row
				MERGE (v:Value {id: row.id})
				SET v.amount = row.amount,
					v.normalized_amount = row.normalized_amount,
					v.year = row.year
				WITH row, v
				MATCH (rw:Row {id: row.row_id})
				MERGE (rw)-[:HAS_VALUE]->(v)
				WITH row, v
				MATCH (r:Report {id: row.report_id})
				MERGE (v)-[:IN_YEAR]->(r)
				""",
				rows=value_rows,
			)

		metric_rows: List[Dict[str, Any]] = []
		metric_links: List[Dict[str, Any]] = []
		for p in payloads:
			for m in p["metrics"]:
				metric_rows.append(m)
			for rel in p["metric_row_links"]:
				metric_links.append(rel)

		if metric_rows:
			tx.run(
				"""
				UNWIND $rows AS row
				MERGE (m:Metric {id: row.id})
				SET m.name = row.name,
					m.category = row.category,
					m.subcategory = row.subcategory
				""",
				rows=metric_rows,
			)

		if metric_links:
			tx.run(
				"""
				UNWIND $rows AS row
				MATCH (m:Metric {id: row.metric_id})
				MATCH (rw:Row {id: row.row_id})
				MERGE (m)-[:MEASURED_BY]->(rw)
				""",
				rows=metric_links,
			)

	@staticmethod
	def _link_next_year_tx(tx: Any) -> None:
		tx.run(
			"""
			MATCH (rw:Row)-[:HAS_VALUE]->(v:Value)
			WHERE v.year IS NOT NULL
			WITH rw, v
			ORDER BY rw.id, v.year
			WITH rw, collect(v) AS vals
			UNWIND range(0, size(vals)-2) AS i
			WITH vals[i] AS a, vals[i+1] AS b
			WHERE a.year < b.year
			MERGE (a)-[:NEXT_YEAR]->(b)
			"""
		)

	def load_batch(self, payloads: List[Dict[str, Any]], dry_run: bool = False) -> GraphLoadStats:
		stats = GraphLoadStats()
		for p in payloads:
			stats.reports += 1
			stats.tables += 1
			stats.rows += len(p["rows"])
			stats.values += len(p["values"])
			stats.chunks += len(p["chunks"])
			stats.metrics += len(p["metrics"])
			stats.companies += 1

		if dry_run:
			return stats

		assert self._driver is not None
		with self._driver.session(database=self.database) as session:
			session.execute_write(self._merge_batch_tx, payloads)

		return stats

	def link_next_year(self, dry_run: bool = False) -> int:
		if dry_run:
			return 0
		assert self._driver is not None
		with self._driver.session(database=self.database) as session:
			session.execute_write(self._link_next_year_tx)
			result = session.run("MATCH ()-[r:NEXT_YEAR]->() RETURN count(r) AS c")
			row = result.single()
			return int(row["c"] if row else 0)

	def validate_integrity(self) -> Dict[str, Any]:
		assert self._driver is not None
		queries = {
			"reports": "MATCH (n:Report) RETURN count(n) AS c",
			"tables": "MATCH (n:Table) RETURN count(n) AS c",
			"rows": "MATCH (n:Row) RETURN count(n) AS c",
			"values": "MATCH (n:Value) RETURN count(n) AS c",
			"chunks": "MATCH (n:TextChunk) RETURN count(n) AS c",
			"metrics": "MATCH (n:Metric) RETURN count(n) AS c",
			"companies": "MATCH (n:Company) RETURN count(n) AS c",
			"orphans_values_without_row": "MATCH (v:Value) WHERE NOT (():Row)-[:HAS_VALUE]->(v) RETURN count(v) AS c",
			"orphans_rows_without_table": "MATCH (r:Row) WHERE NOT (():Table)-[:HAS_ROW]->(r) RETURN count(r) AS c",
			"orphans_tables_without_report": "MATCH (t:Table) WHERE NOT (():Report)-[:CONTAINS_TABLE]->(t) RETURN count(t) AS c",
			"next_year_links": "MATCH ()-[r:NEXT_YEAR]->() RETURN count(r) AS c",
		}

		out: Dict[str, Any] = {}
		with self._driver.session(database=self.database) as session:
			for key, q in queries.items():
				row = session.run(q).single()
				out[key] = int(row["c"] if row else 0)
		return out


def collect_payloads(normalized_dir: str, splits: List[str], limit_per_split: Optional[int] = None) -> List[Dict[str, Any]]:
	payloads: List[Dict[str, Any]] = []
	for split in splits:
		path = os.path.join(normalized_dir, f"finqa_{split}_normalized.jsonl")
		if not os.path.exists(path):
			print(f"skip split={split}: missing {path}")
			continue

		count = 0
		for rec in read_jsonl(path):
			payloads.append(build_graph_payload(rec))
			count += 1
			if limit_per_split is not None and count >= limit_per_split:
				break
		print(f"prepared payloads split={split}, count={count}")
	return payloads


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Batch load normalized FinQA data into Neo4j with idempotent schema mapping")
	parser.add_argument("--normalized-dir", default="data/processed/normalized")
	parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
	parser.add_argument("--limit-per-split", type=int, default=None)
	parser.add_argument("--batch-size", type=int, default=200)
	parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
	parser.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
	parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "password"))
	parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
	parser.add_argument("--max-connection-pool-size", type=int, default=50)
	parser.add_argument("--connection-timeout", type=float, default=30.0)
	parser.add_argument("--max-transaction-retry-time", type=float, default=15.0)
	parser.add_argument("--dry-run", action="store_true")
	return parser.parse_args()


def chunked(items: List[Any], size: int) -> Iterable[List[Any]]:
	step = max(1, size)
	for i in range(0, len(items), step):
		yield items[i : i + step]


def main() -> None:
	args = parse_args()
	payloads = collect_payloads(args.normalized_dir, args.splits, args.limit_per_split)

	loader = GraphLoader(
		uri=args.uri,
		user=args.user,
		password=args.password,
		database=args.database,
		max_connection_pool_size=args.max_connection_pool_size,
		connection_timeout=args.connection_timeout,
		max_transaction_retry_time=args.max_transaction_retry_time,
	)

	if args.dry_run:
		total = GraphLoadStats()
		for batch in chunked(payloads, args.batch_size):
			b = loader.load_batch(batch, dry_run=True)
			total.reports += b.reports
			total.tables += b.tables
			total.rows += b.rows
			total.values += b.values
			total.chunks += b.chunks
			total.metrics += b.metrics
			total.companies += b.companies
		print(
			"dry-run summary: "
			f"reports={total.reports}, tables={total.tables}, rows={total.rows}, "
			f"values={total.values}, chunks={total.chunks}, metrics={total.metrics}, companies={total.companies}"
		)
		return

	loader.connect()
	try:
		loader.create_schema()

		total = GraphLoadStats()
		for batch in chunked(payloads, args.batch_size):
			b = loader.load_batch(batch, dry_run=False)
			total.reports += b.reports
			total.tables += b.tables
			total.rows += b.rows
			total.values += b.values
			total.chunks += b.chunks
			total.metrics += b.metrics
			total.companies += b.companies

		total.next_year_links = loader.link_next_year(dry_run=False)
		integrity = loader.validate_integrity()

		print(
			"load complete: "
			f"reports={total.reports}, tables={total.tables}, rows={total.rows}, values={total.values}, "
			f"chunks={total.chunks}, metrics={total.metrics}, companies={total.companies}, "
			f"next_year_links={total.next_year_links}"
		)
		print(f"integrity: {json.dumps(integrity, sort_keys=True)}")
	finally:
		loader.close()


if __name__ == "__main__":
	main()
