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
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
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

        clean_metric = (metric_name or "").strip().lower().replace("_", " ")
        metric_id = f"metric::{metric_name.strip().lower()}" if metric_name else None

        # WITH ... WHERE ensures optional metric filtering does NOT silently return unrelated rows
        cypher = """
        MATCH (c:Company)
        WHERE c.id = $company_id OR toLower(c.name) CONTAINS $company_slug
        MATCH (c)-[:FILED]->(r:Report)
        WHERE ($record_id IS NULL OR r.record_id = $record_id)
        MATCH (r)-[:CONTAINS_TABLE]->(t:Table)-[:HAS_ROW]->(rw:Row)-[:HAS_VALUE]->(v:Value)
        WHERE ($year IS NULL OR v.year = $year)
        OPTIONAL MATCH (m:Metric)-[:MEASURED_BY]->(rw)
        WITH c, r, rw, v, m
        WHERE ($metric_name IS NULL
               OR m.id = $metric_id
               OR toLower(rw.label) CONTAINS $metric_clean)
        RETURN DISTINCT
            c.name AS company,
            r.record_id AS report_id,
            rw.label AS row_label,
            rw.category AS category,
            v.year AS year,
            v.amount AS amount,
            v.normalized_amount AS normalized_amount
        ORDER BY v.year DESC, rw.label ASC
        LIMIT 50
        """

        params = {
            "company_id": company_id,
            "company_slug": slug_comp,
            "record_id": record_id,
            "year": year,
            "metric_name": metric_name,
            "metric_id": metric_id,
            "metric_clean": clean_metric,
        }

        with driver.session(database=self.database) as session:
            result = session.run(cypher, params)
            records = [
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