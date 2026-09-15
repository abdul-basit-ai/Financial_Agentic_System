"""Structured knowledge graph retrieval tool for Neo4j."""

from __future__ import annotations

import os
import re
from typing import Any

from neo4j import Driver, GraphDatabase
from pydantic import BaseModel, Field

from agent.tools.base import ToolResult

FORBIDDEN_CYPHER_MUTATIONS = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|ALTER|CALL\s+apoc\.periodic)\b",
    re.IGNORECASE,
)


def _slugify(value: str) -> str:
    """Mirrors ingestion.graph_loader.slugify — must stay byte-identical so
    metric/company IDs built here match the IDs written at load time.
    (ingestion cannot import from agent, hence the local copy.)"""
    v = (value or "").strip().lower()
    v = re.sub(r"[^a-z0-9]+", "_", v)
    v = re.sub(r"_+", "_", v).strip("_")
    return v or "unknown"


class GraphQueryInput(BaseModel):
    company_identifier: str = Field(..., description="Company ticker or slug, e.g. 'AAPL' or 'AMZN'")
    metric_name: str | None = Field(None, description="Financial line-item name, e.g. 'revenue' or 'net_income'")
    year: int | None = Field(None, description="Specific fiscal year filter")
    record_id: str | None = Field(None, description="Optional FinQA record identifier filter")


class GraphMetricRecord(BaseModel):
    company: str
    report_id: str
    row_label: str
    category: str
    year: int | None
    amount: float | None
    normalized_amount: float | None


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
        self.uri: str = uri if uri is not None else os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user: str = user if user is not None else os.getenv("NEO4J_USER", "neo4j")
        self.password: str = password if password is not None else os.getenv("NEO4J_PASSWORD", "password")
        self.database = database
        self._driver: Driver | None = None

    def _get_driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def query_metrics(
        self,
        company_identifier: str,
        metric_name: str | None = None,
        year: int | None = None,
        record_id: str | None = None,
    ) -> GraphQueryOutput:
        # Anchor requirement: an unanchored query (UNKNOWN company + no metric
        # + no year + no record_id) matches half the graph via fuzzy CONTAINS
        # and returns arbitrary rows that pollute downstream synthesis.
        if (
            company_identifier.strip().upper() in {"UNKNOWN", "N/A", ""}
            and metric_name is None
            and year is None
            and record_id is None
        ):
            return GraphQueryOutput(
                company_identifier=company_identifier,
                records=[],
                total_found=0,
            )

        driver = self._get_driver()

        slug_comp = company_identifier.strip().lower()
        company_id = f"company::{slug_comp}"
        # Loader writes company IDs from FULL company names ("Amazon Inc" ->
        # company::amazon_inc), while callers usually pass tickers ("AMZN" ->
        # company::amzn). Match both exact forms; the name CONTAINS below is
        # the natural-language fallback.
        company_id_slug = f"company::{_slugify(company_identifier)}"

        clean_metric = (metric_name or "").strip().lower().replace("_", " ").replace("-", " ")
        # Loader writes Metric IDs slugged ("operating income" ->
        # metric::operating_income); building the ID the same way here is what
        # makes the m.id = $metric_id exact path ever fire.
        metric_id = f"metric::{_slugify(metric_name)}" if metric_name else None

        # Query strategy: strictest filter first, then RELAX. FinQA tables are
        # messy (transposed layouts, years as row labels, unparsed header
        # years), so a strict year+metric filter legitimately returns zero rows
        # even when the company has the data. The tool maximizes recall and
        # lets downstream relevance filtering + the grounding gate handle
        # precision. Ladder: metric+year -> year only -> metric only -> company.
        cypher_template = """
        MATCH (c:Company)
        WHERE c.id = $company_id
           OR c.id = $company_id_slug
           OR toLower(c.name) CONTAINS $company_slug
        MATCH (c)-[:FILED]->(r:Report)
        WHERE ($record_id IS NULL OR r.record_id = $record_id)
        MATCH (r)-[:CONTAINS_TABLE]->(t:Table)-[:HAS_ROW]->(rw:Row)-[:HAS_VALUE]->(v:Value)
        WHERE ($year IS NULL OR v.year = $year)
        OPTIONAL MATCH (m:Metric)-[:MEASURED_BY]->(rw)
        WITH c, r, rw, v, m,
             CASE
                 WHEN $metric_name IS NOT NULL AND m.id = $metric_id THEN 0
                 WHEN $metric_name IS NOT NULL
                      AND toLower(replace(rw.label, '-', ' ')) CONTAINS $metric_clean THEN 1
                 ELSE 2
             END AS relevance
        {metric_filter}
        RETURN DISTINCT
            c.name AS company,
            r.record_id AS report_id,
            rw.label AS row_label,
            rw.category AS category,
            v.year AS year,
            v.amount AS amount,
            v.normalized_amount AS normalized_amount,
            relevance
        ORDER BY relevance, v.year DESC, rw.label ASC
        LIMIT 150
        """
        strict_metric = (
            "WHERE ($metric_name IS NULL OR m.id = $metric_id "
            "OR toLower(replace(rw.label, '-', ' ')) CONTAINS $metric_clean)"
        )

        def _run(cypher: str) -> list[GraphMetricRecord]:
            with driver.session(database=self.database) as session:
                result = session.run(cypher, params)
                return [
                    GraphMetricRecord(
                        company=row["company"],
                        report_id=row["report_id"],
                        row_label=row["row_label"],
                        category=row["category"],
                        year=row["year"],
                        amount=row["amount"],
                        normalized_amount=row["normalized_amount"],
                    )
                    for row in result
                ]

        params = {
            "company_id": company_id,
            "company_id_slug": company_id_slug,
            "company_slug": slug_comp,
            "record_id": record_id,
            "year": year,
            "metric_name": metric_name,
            "metric_id": metric_id,
            "metric_clean": clean_metric,
        }

        # Ladder execution: strict metric+year filter first; relax ONE dimension
        # at a time on empty results (metric -> year -> both) so a company with
        # data never returns zero rows just because label/year metadata was
        # unparseable for the gold table.
        records: list[GraphMetricRecord] = _run(cypher_template.format(metric_filter=strict_metric))
        if not records and metric_name is not None:
            records = _run(cypher_template.format(metric_filter="WHERE $metric_name IS NULL"))
        if not records and year is not None:
            records = _run(cypher_template.format(metric_filter=strict_metric).replace(
                "WHERE ($year IS NULL OR v.year = $year)",
                "WHERE 1 = 1",
            ))
        if not records:
            records = _run(cypher_template.format(metric_filter="WHERE 1 = 1"))

        return GraphQueryOutput(
            company_identifier=company_identifier,
            records=records,
            total_found=len(records),
        )

    def execute_safe_cypher(self, cypher_query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Executes custom read-only Cypher queries, barring mutations."""
        if FORBIDDEN_CYPHER_MUTATIONS.search(cypher_query):
            raise PermissionError("Write or schema-modifying Cypher statements are strictly prohibited")

        driver = self._get_driver()
        with driver.session(database=self.database) as session:
            result = session.run(cypher_query, params)
            return [dict(record) for record in result]


def graph_retrieval_tool(
    payload: GraphQueryInput, client: GraphRetrievalTool | None = None
) -> ToolResult[GraphQueryOutput]:
    """Instrumented tool entrypoint for structured graph retrieval."""
    instance = client or GraphRetrievalTool()
    return ToolResult.execute_instrumented(
        tool_name="graph_retrieval",
        fn=instance.query_metrics,
        company_identifier=payload.company_identifier,
        metric_name=payload.metric_name,
        year=payload.year,
        record_id=payload.record_id,
    )
