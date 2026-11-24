"""Full-table extraction tool for the anchored FinQA filing.

Table-conditioned program generation needs the ENTIRE table — every row
crossed with every column header — not retrieved value fragments. FinQA
tables are small (5-15 rows x 3-6 columns), so one anchored Cypher query
returns the complete reasoning surface in a single envelope.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent.tools.base import ToolResult
from agent.tools.graph_tool import GraphRetrievalTool


class TableExtractInput(BaseModel):
    record_id: str = Field(
        ...,
        description="FinQA record id of the filing to extract, e.g. 'V/2008/page_17.pdf-1'",
    )


class TableCell(BaseModel):
    column_index: int
    column_header: str
    amount: float | None
    year: int | None = None
    normalized_amount: float | None = None
    evidence_id: str | None = None
    table_id: str | None = None
    page: int | None = None
    raw: str | None = None
    unit_label: str | None = None
    currency: str | None = None
    quantity_kind: str | None = None
    is_percentage: bool | None = None
    scale_multiplier: float | None = None
    units_ambiguous: bool | None = None
    is_missing: bool | None = None


class TableRow(BaseModel):
    row_index: int
    row_label: str
    cells: list[TableCell]
    evidence_id: str | None = None
    table_id: str | None = None
    page: int | None = None
    gold_id: str | None = None


class TableExtractOutput(BaseModel):
    record_id: str
    headers: list[str]
    rows: list[TableRow]
    total_rows: int
    total_values: int
    source_file: str | None = None


class TableExtractionTool:
    """Reads the complete table of one filing from Neo4j."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str = "neo4j",
        graph_client: GraphRetrievalTool | None = None,
    ) -> None:
        self._owns_client = graph_client is None
        self._graph = graph_client or GraphRetrievalTool(uri, user, password, database)

    def close(self) -> None:
        if self._owns_client:
            self._graph.close()

    def extract(self, record_id: str) -> TableExtractOutput:
        cypher = """
        MATCH (r:Report {record_id: $record_id})-[:CONTAINS_TABLE]->(t:Table)
        OPTIONAL MATCH (t)-[:HAS_ROW]->(rw:Row)
        OPTIONAL MATCH (rw)-[:HAS_VALUE]->(v:Value)
        RETURN r.source_file AS source_file, r.page AS page,
               t.id AS table_id, t.header AS headers,
               rw.source_row_index AS row_index, rw.label AS row_label,
               rw.evidence_id AS row_evidence_id, rw.gold_id AS gold_id,
               properties(v) AS cell
        ORDER BY table_id, row_index, v.column_index
        """
        result = self._graph.read_rows(cypher, {"record_id": record_id})
        rows: dict[tuple[str, int], TableRow] = {}
        headers = result[0]["headers"] or [] if result else []
        total_values = 0
        for record in result:
            ridx = record["row_index"]
            if ridx is None:
                continue
            table_id = record["table_id"]
            key = table_id, ridx
            if key not in rows:
                rows[key] = TableRow(
                    table_id=table_id,
                    row_index=ridx,
                    row_label=str(record["row_label"] or ""),
                    evidence_id=record["row_evidence_id"],
                    gold_id=record["gold_id"],
                    page=record["page"],
                    cells=[],
                )
            cell = record["cell"]
            if cell is not None:
                rows[key].cells.append(
                    TableCell.model_validate(
                        {**cell, "table_id": table_id, "page": record["page"]}
                    )
                )
                total_values += cell.get("amount") is not None

        return TableExtractOutput(
            record_id=record_id,
            headers=headers,
            rows=list(rows.values()),
            total_rows=len(rows),
            total_values=total_values,
            source_file=result[0]["source_file"] if result else None,
        )


def table_extract_tool(
    payload: TableExtractInput, client: TableExtractionTool | None = None
) -> ToolResult[TableExtractOutput]:
    """Instrumented tool entrypoint for full-table extraction."""
    from agent.tools._clients import get_table_tool_client

    instance: TableExtractionTool = client or get_table_tool_client()
    return ToolResult.execute_instrumented(
        tool_name="table_extract",
        fn=instance.extract,
        record_id=payload.record_id,
    )
