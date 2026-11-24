"""Structured knowledge graph retrieval tool for Neo4j."""

from __future__ import annotations

import os
import re
from threading import Lock
from typing import Any

from neo4j import READ_ACCESS, Driver, GraphDatabase, Query
from pydantic import BaseModel, Field

from agent.tools.base import ToolResult
from ingestion.graph_loader import slugify as _slugify

FORBIDDEN_CYPHER_MUTATIONS = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|ALTER|CALL\s+apoc\.periodic)\b",
    re.IGNORECASE,
)


class GraphQueryInput(BaseModel):
    company_identifier: str = Field(
        ..., description="Company ticker or slug, e.g. 'AAPL' or 'AMZN'"
    )
    metric_name: str | None = Field(
        None, description="Financial line-item name, e.g. 'revenue' or 'net_income'"
    )
    year: int | None = Field(None, description="Specific fiscal year filter")
    record_id: str | None = Field(
        None, description="Optional FinQA record identifier filter"
    )


class GraphMetricRecord(BaseModel):
    company: str
    report_id: str
    row_label: str
    category: str
    year: int | None
    amount: float | None
    normalized_amount: float | None
    column_index: int | None = None
    column_header: str | None = None
    evidence_id: str | None = None
    table_id: str | None = None
    row_index: int | None = None
    row_evidence_id: str | None = None
    source_file: str | None = None
    page: int | None = None
    raw: str | None = None
    unit_label: str | None = None
    currency: str | None = None
    quantity_kind: str | None = None
    is_percentage: bool | None = None
    scale_multiplier: float | None = None
    units_ambiguous: bool | None = None
    gold_id: str | None = None


class GraphQueryOutput(BaseModel):
    company_identifier: str
    records: list[GraphMetricRecord]
    total_found: int


class GraphRetrievalTool:
    """Manages Neo4j queries with injection defense and parameterization."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str = "neo4j",
    ) -> None:
        self.uri: str = (
            uri if uri is not None else os.getenv("NEO4J_URI", "bolt://localhost:7687")
        )
        self.user: str = user if user is not None else os.getenv("NEO4J_USER", "neo4j")
        self.password: str = (
            password
            if password is not None
            else os.getenv("NEO4J_PASSWORD", "password")
        )
        self.database = database
        self._driver: Driver | None = None
        self._driver_lock = Lock()

    def _get_driver(self) -> Driver:
        with self._driver_lock:
            if self._driver is None:
                self._driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password),
                    connection_timeout=3,
                    connection_acquisition_timeout=5,
                )
            return self._driver

    def close(self) -> None:
        with self._driver_lock:
            if self._driver is not None:
                self._driver.close()
                self._driver = None

    def read_rows(self, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        with self._get_driver().session(
            database=self.database, default_access_mode=READ_ACCESS
        ) as session:
            return [
                record.data()
                for record in session.run(Query(cypher, timeout=15), params)
            ]

    def query_metrics(
        self,
        company_identifier: str,
        metric_name: str | None = None,
        year: int | None = None,
        record_id: str | None = None,
    ) -> GraphQueryOutput:
        company = company_identifier.strip().lower()
        if company in {"unknown", "n/a", ""} and not record_id:
            return GraphQueryOutput(
                company_identifier=company_identifier, records=[], total_found=0
            )

        params = {
            "company_ids": [
                f"company::{company}",
                f"company::{_slugify(company_identifier)}",
            ],
            "company_name": company,
            "record_id": record_id,
            "year": year,
            "metric_id": f"metric::{_slugify(metric_name)}" if metric_name else None,
            "metric_words": _slugify(metric_name).split("_") if metric_name else [],
        }
        cypher = """
        MATCH (c:Company)-[:FILED]->(r:Report)
        WHERE ($record_id IS NOT NULL AND r.record_id = $record_id)
           OR ($record_id IS NULL AND (c.id IN $company_ids OR toLower(c.name) = $company_name))
        MATCH (r)-[:CONTAINS_TABLE]->(t:Table)-[:HAS_ROW]->(rw:Row)-[:HAS_VALUE]->(v:Value)
        WHERE v.amount IS NOT NULL AND ($year IS NULL OR v.year = $year)
        WITH c, r, t, rw, v,
             EXISTS { MATCH (m:Metric {id: $metric_id})-[:MEASURED_BY]->(rw) } AS exact_metric
        WHERE exact_metric OR all(w IN $metric_words
            WHERE toLower(rw.label) CONTAINS w OR toLower(coalesce(v.column_header, '')) CONTAINS w)
        RETURN c.name AS company, r.record_id AS report_id, r.source_file AS source_file,
               r.page AS page, t.id AS table_id,
               rw.label AS row_label, rw.category AS category, rw.source_row_index AS row_index,
               rw.evidence_id AS row_evidence_id, rw.gold_id AS gold_id,
               v.year AS year, v.amount AS amount, v.normalized_amount AS normalized_amount,
               v.column_index AS column_index, v.column_header AS column_header,
               v.evidence_id AS evidence_id, v.raw AS raw, v.unit_label AS unit_label,
               v.quantity_kind AS quantity_kind, v.currency AS currency,
               v.scale_multiplier AS scale_multiplier, v.is_percentage AS is_percentage,
               v.units_ambiguous AS units_ambiguous
        ORDER BY exact_metric DESC, year DESC, report_id, row_index, column_index
        LIMIT 150
        """
        records = [
            GraphMetricRecord.model_validate(row)
            for row in self.read_rows(cypher, params)
        ]
        return GraphQueryOutput(
            company_identifier=company_identifier,
            records=records,
            total_found=len(records),
        )

    def execute_safe_cypher(
        self, cypher_query: str, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Executes custom read-only Cypher queries, barring mutations."""
        if FORBIDDEN_CYPHER_MUTATIONS.search(cypher_query):
            raise PermissionError(
                "Write or schema-modifying Cypher statements are strictly prohibited"
            )

        return self.read_rows(cypher_query, params)


def list_company_record_ids(
    company_identifier: str, client: GraphRetrievalTool | None = None
) -> list[str]:
    """All FinQA record ids filed by a company (cheap Cypher, Neo4j).

    Feeds record anchoring: the vector search is restricted to these ids so
    the top chunk identifies the filing the question is actually about.
    Degrades to an empty list when the company is unanchored or Neo4j fails —
    callers then fall back to unscoped vector search.
    """
    slug = (company_identifier or "").strip().lower()
    if slug in {"", "unknown", "n/a"}:
        return []
    from agent.tools._clients import get_graph_tool_client

    instance: GraphRetrievalTool = client or get_graph_tool_client()
    cypher = """
    MATCH (c:Company)-[:FILED]->(r:Report)
    WHERE c.id IN $company_ids OR toLower(c.name) = $company_name
    RETURN DISTINCT r.record_id AS record_id
    ORDER BY record_id
    """
    try:
        rows = instance.read_rows(
            cypher,
            {
                "company_ids": [
                    f"company::{slug}",
                    f"company::{_slugify(company_identifier)}",
                ],
                "company_name": slug,
            },
        )
        return [str(row["record_id"]) for row in rows if row["record_id"]]
    except Exception:
        return []


def graph_retrieval_tool(
    payload: GraphQueryInput, client: GraphRetrievalTool | None = None
) -> ToolResult[GraphQueryOutput]:
    """Instrumented tool entrypoint for structured graph retrieval."""
    from agent.tools._clients import get_graph_tool_client

    instance: GraphRetrievalTool = client or get_graph_tool_client()
    return ToolResult.execute_instrumented(
        tool_name="graph_retrieval",
        fn=instance.query_metrics,
        company_identifier=payload.company_identifier,
        metric_name=payload.metric_name,
        year=payload.year,
        record_id=payload.record_id,
    )
