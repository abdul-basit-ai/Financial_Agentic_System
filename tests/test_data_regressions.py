from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from agent.tools import _clients, vector_tool
from agent.tools.graph_tool import GraphRetrievalTool
from agent.tools.table_tool import TableExtractionTool
from ingestion.graph_loader import build_graph_payload
from ingestion.normalizer import normalize_cell, normalize_record, normalize_table
from ingestion.parser import parse_record
from ingestion.table_parser import parse_table_structure
from ingestion.text_chunker import chunk_context


def test_financial_cells_keep_units_scoped() -> None:
    rows = normalize_table(
        [
            ["Statement (in USD thousands, except per share data)", "2021", "2020"],
            ["Revenue (in millions)", "1,200", "60"],
            ["Debt (in billions)", "3", "EUR 2 billion"],
            ["Shares outstanding in millions", "4", "5"],
            ["Diluted EPS", "$(1.25)", "$0.75"],
            ["Operating margin", "12.5%", "(10%)"],
            ["Operating cash flows", "7", "5"],
        ],
        text_context="The company earned 2 million from an unrelated investment.",
    )["rows"]
    revenue, debt, shares, eps, margin, cash = [row[1:] for row in rows]
    assert revenue[0]["numeric_value_base"] == 1_200_000_000
    assert debt[0]["numeric_value_base"] == 3_000_000_000
    assert debt[1]["currency"] == "EUR"
    assert shares[0]["numeric_value_base"] == 4_000_000
    assert shares[0]["quantity_kind"] == "shares"
    assert shares[0]["currency"] is None
    assert eps[0]["quantity_kind"] == "per_share"
    assert eps[0]["currency"] == "USD"
    assert eps[0]["numeric_value_base"] == -1.25
    assert margin[0]["numeric_value_base"] == 12.5
    assert margin[1]["numeric_value_base"] == -10
    assert margin[1]["is_percentage"]
    assert cash[0]["numeric_value_base"] == 7_000


def test_cell_override_and_ambiguous_declarations() -> None:
    cell = normalize_cell("  CAD (3.25 billion)  ", "in USD millions")
    assert cell["raw"] == "  CAD (3.25 billion)  "
    assert cell["numeric_value"] == -3.25
    assert cell["numeric_value_base"] == -3_250_000_000
    assert cell["currency"] == "CAD"

    conflicting = normalize_cell("2", "in millions or in billions")
    assert conflicting["units_ambiguous"]
    assert conflicting["numeric_value_base"] is None
    assert conflicting["scale_multiplier"] is None


def test_two_column_rows_and_ragged_headers_preserve_source_coordinates() -> None:
    table = [
        ["Year", "2021"],
        ["Revenue", "100"],
        ["Debt", "22"],
        ["Shares outstanding", "35"],
    ]
    assert parse_table_structure(table)["data_start_row"] == 1
    assert normalize_table(table)["source_row_indices"] == [1, 2, 3]
    ragged = [
        ["Metric"],
        ["", "Year Ended 2021", "Year Ended 2020"],
        ["Debt", "3", "2"],
    ]
    assert parse_table_structure(ragged)["data_start_row"] == 2
    assert normalize_table(ragged)["source_row_indices"] == [2]
    assert normalize_table(ragged)["header"] == [
        "Metric",
        "Year Ended 2021",
        "Year Ended 2020",
    ]


def test_graph_payload_and_table_tool_provenance() -> None:
    record = normalize_record(
        parse_record(
            "dev",
            {
                "id": "XYZ/2021/page_39.pdf-3",
                "filename": "XYZ/2021/page_39.pdf",
                "table": [
                    ["Balance sheet", "2021", "2020"],
                    ["Debt in billions", "3", "-"],
                ],
            },
        )
    )
    payload = build_graph_payload(record)
    amount, missing = payload["values"]
    row = payload["rows"][0]
    assert payload["report"]["page"] == 39
    assert row["source_row_index"] == 1
    assert row["gold_id"] == "table_1"
    assert amount["normalized_amount"] == 3_000_000_000
    assert amount["evidence_id"] == f"{row['evidence_id']}::col::1"
    assert missing["is_missing"]

    client = MagicMock(spec=GraphRetrievalTool)
    client.read_rows.return_value = [
        {
            "source_file": payload["report"]["source_file"],
            "table_id": payload["table"]["id"],
            "headers": payload["table"]["header"],
            "row_index": row["source_row_index"],
            "row_label": row["label"],
            "row_evidence_id": row["evidence_id"],
            "gold_id": row["gold_id"],
            "page": payload["report"]["page"],
            "cell": cell,
        }
        for cell in (amount, missing)
    ]
    table = TableExtractionTool(graph_client=client).extract(record["record_id"])
    assert table.headers == ["Balance sheet", "2021", "2020"]
    assert table.total_values == 1
    assert table.rows[0].cells[0].scale_multiplier == 1_000_000_000
    assert table.rows[0].cells[1].raw == "-"


def test_chunks_keep_text_indices_across_blank_lines() -> None:
    chunks = chunk_context(
        ["A long sentence.", "", "Another long sentence."], ["After the table."]
    )
    assert (chunks[0]["start_sentence"], chunks[0]["end_sentence"]) == (0, 2)
    assert (chunks[1]["start_sentence"], chunks[1]["end_sentence"]) == (3, 3)


def test_vector_scoped_query_recovers_after_failed_sql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = MagicMock()
    encoder.encode.return_value = [1.0] + [0.0] * 383
    monkeypatch.setenv("EMBEDDING_MODEL", "fixture-encoder")
    monkeypatch.setattr(_clients, "get_sentence_encoder", lambda name: encoder)
    connection_args: list[dict[str, Any]] = []
    transactions = [MagicMock(), MagicMock()]

    def connect(**kwargs: Any) -> Any:
        connection_args.append(kwargs)
        return transactions[len(connection_args) - 1]

    monkeypatch.setattr(vector_tool.psycopg2, "connect", connect)
    first_cursor, second_cursor = [
        tx.cursor.return_value.__enter__.return_value for tx in transactions
    ]
    first_cursor.execute.side_effect = RuntimeError("aborted transaction")
    second_cursor.fetchall.return_value = [
        (
            "XYZ/2021/page_39.pdf-1",
            "XYZ/2021/page_39.pdf",
            "pre_text",
            1,
            "Net income was 20.",
            0.91,
            2,
            4,
            "2",
        )
    ]
    client = vector_tool.VectorRetrievalTool()
    kwargs = {"record_ids": ["XYZ/2021/page_39.pdf-1"], "company_identifier": "XYZ"}
    with pytest.raises(RuntimeError, match="aborted transaction"):
        client.search("Net income", **kwargs)
    result = client.search("Net income", **kwargs)
    for tx in transactions:
        tx.close.assert_called_once()
    assert transactions[0].__exit__.call_args.args[0] is RuntimeError
    assert all(args["connect_timeout"] == 3 for args in connection_args)
    sql, params = second_cursor.execute.call_args.args
    assert "scoped AS MATERIALIZED" in sql
    assert "FROM scoped" in sql
    assert params["record_ids"] == kwargs["record_ids"]
    assert params["company"] == "XYZ"
    assert params["model"] == "fixture-encoder"
    assert result.chunks[0].gold_ids == ["text_2", "text_3", "text_4"]
    assert result.chunks[0].page == 39
