"""Phase 3 Knowledge Graph & Vector Store Integrity Suite.

Validates:
1. Entity extraction does not pollute company names with document tokens.
2. Table parser preserves rows with currency symbols and accounting negatives.
3. Metric matching isolates whole words and prevents false positives.
4. Neo4j batch loading adheres to relational tree contracts without orphan values.
5. pgvector indexes text chunks and executes cosine similarity queries.
"""

from __future__ import annotations

import os
import pytest

from ingestion.entity_extractor import extract_entities
from ingestion.graph_loader import _match_metric, build_graph_payload, GraphLoader
from ingestion.table_parser import parse_table_structure
from ingestion.text_chunker import chunk_context
from knowledge.vector_store.loader import VectorStoreLoader


# =====================================================================
# Unit Validations (Zero DB Dependency)
# =====================================================================


def test_table_parser_preserves_currency_and_negative_rows() -> None:
    raw_table = [
        ["Line Item", "2020", "2019"],
        ["Operating Revenue", "$1,200", "$1,100"],
        ["Net Loss", "$(150)", "$(120)"],
        ["Tax Rate", "21%", "20%"],
    ]
    parsed = parse_table_structure(raw_table)
    assert parsed["header_depth"] == 1
    assert parsed["data_start_row"] == 1
    assert parsed["row_count"] == 4


def test_entity_extractor_rejects_page_and_doc_identifiers() -> None:
    entities = extract_entities(
        question="What was net revenue?",
        filename="page_45.pdf-1.json",
        pre_text=["The company recorded profit."],
        post_text=[],
        table=[["Metric", "2020"], ["Revenue", "100"]],
    )
    assert "PAGE" not in entities["company_names"]
    assert "DOC" not in entities["company_names"]


def test_entity_extractor_captures_valid_company_tickers() -> None:
    entities = extract_entities(
        question="What was operating margin?",
        filename="AAPL/2020/page_12.json",
        pre_text=["Apple Inc. filed Form 10-K."],
        post_text=[],
        table=[],
    )
    assert "AAPL" in entities["company_names"]


def test_metric_matching_prevents_false_positives() -> None:
    # Exact and bounded phrases match
    assert _match_metric("revenue", "Total revenue")
    assert _match_metric("operating_income", "Operating income (loss)")
    assert _match_metric("tax", "Income tax expense")

    # Partial-word and reverse substring inclusions must NOT match
    assert not _match_metric("tax", "Pretax income")
    assert not _match_metric("other operating expenses", "Other")
    assert not _match_metric("net", "Internet services")


def test_graph_payload_generation_integrity() -> None:
    mock_record = {
        "record_id": "test_rec_001",
        "split": "train",
        "document": {
            "filename": "MSFT/2021/page_1.json",
            "question": "What was total revenue in 2021?",
        },
        "entities": {
            "company_names": ["MSFT"],
            "fiscal_years": ["2021", "2020"],
            "metric_names_resolved": ["revenue"],
        },
        "context": {
            "chunks": [{"text": "Microsoft Corporation posted strong growth."}],
        },
        "table": {
            "header": ["Metric", "2021", "2020"],
            "rows": [
                [
                    {"raw": "Total Revenue"},
                    {"numeric_value": 168000.0, "numeric_value_base": 168000000000.0},
                    {"numeric_value": 143000.0, "numeric_value_base": 143000000000.0},
                ]
            ],
        },
    }

    payload = build_graph_payload(mock_record)

    assert payload["company"]["id"] == "company::msft"
    assert payload["report"]["id"] == "report::test_rec_001"
    assert len(payload["rows"]) == 1
    assert len(payload["values"]) == 2

    val_2021 = next(v for v in payload["values"] if v["year"] == 2021)
    assert val_2021["amount"] == 168000.0
    assert val_2021["normalized_amount"] == 168000000000.0
    assert val_2021["row_id"] == "row::test_rec_001::0"


def test_text_chunker_overlapping_windows() -> None:
    sentences = [f"Sentence {i}." for i in range(5)]
    chunks = chunk_context(pre_text=sentences, post_text=[], chunk_size=3, stride=2)

    assert len(chunks) == 2
    assert chunks[0]["start_sentence"] == 0
    assert chunks[0]["end_sentence"] == 2
    assert chunks[1]["start_sentence"] == 2
    assert chunks[1]["end_sentence"] == 4


# =====================================================================
# Database Integration Tests (Requires Running Compose Services)
# =====================================================================


@pytest.mark.integration
def test_neo4j_batch_load_and_integrity() -> None:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    loader = GraphLoader(uri=uri, user=user, password=password)
    try:
        loader.connect()
    except Exception as exc:
        pytest.skip(f"Neo4j container not available: {exc}")

    loader.create_schema()

    payloads = [
        build_graph_payload({
            "record_id": "test_id_1",
            "split": "train",
            "document": {"filename": "AMZN/2020/p1.json", "question": "Q1"},
            "entities": {"company_names": ["AMZN"], "fiscal_years": ["2020"]},
            "table": {"header": ["Line", "2020"], "rows": [[{"raw": "Sales"}, {"numeric_value": 100.0}]]},
        }),
        build_graph_payload({
            "record_id": "test_id_2",
            "split": "train",
            "document": {"filename": "AMZN/2020/p2.json", "question": "Q2"},
            "entities": {"company_names": ["AMZN"], "fiscal_years": ["2020"]},
            "table": {"header": ["Line", "2020"], "rows": [[{"raw": "Profit"}, {"numeric_value": 10.0}]]},
        }),
    ]

    stats = loader.load_batch(payloads)
    assert stats.reports == 2
    assert stats.values == 2

    integrity = loader.validate_integrity()
    assert integrity["orphans_values_without_row"] == 0
    assert integrity["orphans_rows_without_table"] == 0
    assert integrity["orphans_tables_without_report"] == 0

    loader.close()


@pytest.mark.integration
def test_pgvector_load_and_search() -> None:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    dbname = os.getenv("POSTGRES_DB", "financial_agent")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "password")

    v_loader = VectorStoreLoader(host=host, port=port, dbname=dbname, user=user, password=password)
    try:
        v_loader.connect()
    except Exception as exc:
        pytest.skip(f"Postgres container not available: {exc}")

    v_loader.create_schema()

    mock_rec = {
        "record_id": "test_vec_rec_001",
        "split": "train",
        "document": {"filename": "AAPL/2020/p1.json"},
        "context": {
            "chunks": [
                {"source": "pre_text", "text": "Operating revenues climbed 15% due to high iPhone sales."},
                {"source": "post_text", "text": "Research and development costs remained stable."},
            ]
        },
    }

    rows = v_loader.load_chunks_from_record(mock_rec)
    assert len(rows) == 2
    inserted = v_loader.insert_batch(rows)
    assert inserted == 2

    results = v_loader.query_similarity("iPhone sales and revenue", top_k=1)
    assert len(results) > 0
    assert results[0]["record_id"] == "test_vec_rec_001"
    assert "iPhone" in results[0]["text_content"]

    v_loader.close()