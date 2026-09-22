"""Full-table extraction tool for the anchored FinQA filing.

Table-conditioned program generation needs the ENTIRE table — every row
crossed with every column header — not retrieved value fragments. FinQA
tables are small (5-15 rows x 3-6 columns), so one anchored Cypher query
returns the complete reasoning surface in a single envelope.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from agent.tools.base import ToolResult


class TableExtractInput(BaseModel):
    record_id: str = Field(..., description="FinQA record id of the filing to extract, e.g. 'V/2008/page_17.pdf-1'")


class TableCell(BaseModel):
    column_index: int
    column_header: str
    amount: float | None
    year: int | None = None


class TableRow(BaseModel):
    row_index: int
    row_label: str
    cells: list[TableCell]


class TableExtractOutput(BaseModel):
    record_id: str
    headers: list[str]
    rows: list[TableRow]
    total_rows: int
    total_values: int


class TableExtractionTool:
    """Reads the complete table of one filing from Neo4j."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str = "neo4j",
    ) -> None:
        self.uri: str = uri if uri is not None else os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user: str = user if user is not None else os.getenv("NEO4J_USER", "neo4j")
        self.password: str = (
            password if password is not None else os.getenv("NEO4J_PASSWORD", "password")
        )
        self.database = database
        self._driver: Any = None

    def _get_driver(self) -> Any:
        if self._driver is None:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def extract(self, record_id: str) -> TableExtractOutput:
        driver = self._get_driver()
        cypher = """
        MATCH (r:Report {record_id: $record_id})-[:CONTAINS_TABLE]->(t:Table)
        OPTIONAL MATCH (t)-[:HAS_ROW]->(rw:Row)
        OPTIONAL MATCH (rw)-[:HAS_VALUE]->(v:Value)
        RETURN rw.source_row_index AS row_index, rw.label AS row_label,
               v.column_index AS col_index, v.column_header AS col_header,
               v.amount AS amount, v.year AS year
        ORDER BY row_index, col_index
        """
        with driver.session(database=self.database) as session:
            result = session.run(cypher, {"record_id": record_id})
            rows: list[TableRow] = []
            headers: list[str] = []
            total_values = 0
            current: TableRow | None = None
            for record in result:
                header = record["col_header"]
                if header and header not in headers:
                    headers.append(header)
                ridx = record["row_index"]
                if ridx is None:
                    continue
                if current is None or current.row_index != ridx:
                    current = TableRow(
                        row_index=ridx, row_label=str(record["row_label"] or ""), cells=[]
                    )
                    rows.append(current)
                if record["col_index"] is not None:
                    current.cells.append(
                        TableCell(
                            column_index=record["col_index"],
                            column_header=str(header or ""),
                            amount=record["amount"],
                            year=record["year"],
                        )
                    )
                    total_values += 1

        return TableExtractOutput(
            record_id=record_id,
            headers=headers,
            rows=rows,
            total_rows=len(rows),
            total_values=total_values,
        )


def table_extract_tool(
    payload: TableExtractInput, client: TableExtractionTool | None = None
) -> ToolResult[TableExtractOutput]:
    """Instrumented tool entrypoint for full-table extraction."""
    instance = client or TableExtractionTool()
    return ToolResult.execute_instrumented(
        tool_name="table_extract",
        fn=instance.extract,
        record_id=payload.record_id,
    )
