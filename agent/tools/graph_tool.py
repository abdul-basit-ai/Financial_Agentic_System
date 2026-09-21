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

    def _get_driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
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

        clean_metric = (
            (metric_name or "").strip().lower().replace("_", " ").replace("-", " ")
        )
        # Word-level matching: questions use singular forms ("payment volume")
        # while headers store plurals ("payments volume ( billions )"). Full
        # -phrase CONTAINS fails on that 's'; per-word substring matching
        # survives it ("payment" ⊂ "payments"). Row label OR column header.
        metric_words = [w for w in clean_metric.split() if w]
        # Loader writes Metric IDs slugged ("operating income" ->
        # metric::operating_income); building the ID the same way here is what
        # makes the m.id = $metric_id exact path ever fire.
        metric_id = f"metric::{_slugify(metric_name)}" if metric_name else None

        # Query strategy: strictest filter first, then RELAX. FinQA tables are
        # messy (transposed layouts, years as row labels, unparsed header
        # years), so a strict year+metric filter legitimately returns zero rows
        # even when the company has the data. The tool maximizes recall and
        # lets downstream relevance filtering + the grounding gate handle
        # precision.
        #
        # Two shapes:
        # - ANCHORED (record_id set): match the Report directly. The record id
        #   is the filing the question is about and is NEVER relaxed — a
        #   company-wide fallback re-introduces the cross-filing duplicate
        #   flood that buried gold rows under identical labels from other
        #   filings' tables.
        # - UNANCHORED: company predicate + filters, relaxed per the ladder.
        #
        # Year matching treats v.year IS NULL as a wildcard: ~41% of Values
        # have no parseable header year (transposed tables, metric-name
        # headers), and strict equality silently excluded all of them.
        def _build_cypher(anchored: bool, metric_filter: str, year_filter: str) -> str:
            if anchored:
                match_clause = """
        MATCH (c:Company)-[:FILED]->(r:Report {record_id: $record_id})
        MATCH (r)-[:CONTAINS_TABLE]->(t:Table)-[:HAS_ROW]->(rw:Row)-[:HAS_VALUE]->(v:Value)"""
            else:
                match_clause = """
        MATCH (c:Company)
        WHERE c.id = $company_id
           OR c.id = $company_id_slug
           OR toLower(c.name) CONTAINS $company_slug
        MATCH (c)-[:FILED]->(r:Report)
        MATCH (r)-[:CONTAINS_TABLE]->(t:Table)-[:HAS_ROW]->(rw:Row)-[:HAS_VALUE]->(v:Value)"""
            return f"""{match_clause}
        WHERE {year_filter} AND {metric_filter}
        OPTIONAL MATCH (m:Metric)-[:MEASURED_BY]->(rw)
        WITH c, r, rw, v, m,
             CASE
                 WHEN $metric_name IS NOT NULL AND m.id = $metric_id THEN -1000
                 ELSE -(size([w IN $metric_words
                       WHERE toLower(rw.label) CONTAINS w
                          OR toLower(coalesce(v.column_header, '')) CONTAINS w]))
             END AS relevance
        RETURN DISTINCT
            c.name AS company,
            r.record_id AS report_id,
            rw.label AS row_label,
            rw.category AS category,
            v.year AS year,
            v.amount AS amount,
            v.normalized_amount AS normalized_amount,
            v.column_index AS column_index,
            v.column_header AS column_header,
            relevance
        ORDER BY relevance, v.year DESC, rw.label ASC
        LIMIT 150
        """

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
                        column_index=row["column_index"],
                        column_header=row["column_header"],
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
            "metric_words": metric_words,
        }

        # Filters are single WHERE predicates (year AND metric combined by the
        # builder). "Relaxed" metric filter must be a neutral predicate:
        # $metric_name is still bound (non-NULL) in params, so a
        # `$metric_name IS NULL` rung would filter out every row. `true` keeps
        # the rows while the relevance CASE still ranks label/header matches
        # first.
        no_metric = "true"
        # Self-contained on rw/v: WHERE runs before the OPTIONAL MATCH that
        # defines `m`, so the metric-id equality path lives only in the
        # relevance CASE. Word-level matching: every hint word must appear in
        # the row label or column header.
        strict_metric = (
            "($metric_name IS NULL OR size($metric_words) = 0 OR "
            "all(w IN $metric_words WHERE "
            "toLower(rw.label) CONTAINS w "
            "OR toLower(coalesce(v.column_header, '')) CONTAINS w))"
        )
        year_where = "($year IS NULL OR v.year IS NULL OR v.year = $year)"
        no_year = "true"

        records: list[GraphMetricRecord] = []
        if record_id is not None:
            # Anchored ladder: relax metric, then year — never the record.
            for metric_filter, year_filter in [
                (strict_metric, year_where),
                (no_metric, year_where),
                (strict_metric, no_year),
                (no_metric, no_year),
            ]:
                records = _run(_build_cypher(True, metric_filter, year_filter))
                if records:
                    break
            return GraphQueryOutput(
                company_identifier=company_identifier,
                records=records,
                total_found=len(records),
            )

        # Unanchored ladder: metric -> year -> company-wide sweep.
        for metric_filter, year_filter in [
            (strict_metric, year_where),
            (no_metric, year_where),
            (no_metric, no_year),
        ]:
            records = _run(_build_cypher(False, metric_filter, year_filter))
            if records:
                break

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

        driver = self._get_driver()
        with driver.session(database=self.database) as session:
            result = session.run(cypher_query, params)
            return [dict(record) for record in result]


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
    instance = client or GraphRetrievalTool()
    cypher = """
    MATCH (c:Company)-[:FILED]->(r:Report)
    WHERE c.id = $company_id OR c.id = $company_id_slug
       OR toLower(c.name) CONTAINS $company_slug
    RETURN DISTINCT r.record_id AS record_id
    """
    try:
        with instance._get_driver().session(database=instance.database) as session:
            return [
                row["record_id"]
                for row in session.run(
                    cypher,
                    {
                        "company_id": f"company::{slug}",
                        "company_id_slug": f"company::{_slugify(company_identifier)}",
                        "company_slug": slug,
                    },
                )
                if row["record_id"]
            ]
    except Exception:
        return []


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
