import json
from pathlib import Path

from ingestion import parser as finqa_parser
from ingestion.entity_extractor import extract_entities
from ingestion.normalizer import normalize_table


def test_parse_record_extracts_core_fields():
    raw = {
        "id": "abc123",
        "filename": "sample_10k.txt",
        "pre_text": ["Company generated strong cash flow in 2020."],
        "post_text": ["Net income increased year over year."],
        "table": [["Metric", "2020", "2019"], ["Net income (loss)", "100", "90"]],
        "qa": {
            "question": "What is net income in 2020?",
            "answer": "100",
            "exe_ans": 100.0,
            "program": "subtract(100, 90)",
            "program_re": "subtract(100, 90)",
            "gold_inds": {"table_1": "..."},
        },
    }

    rec = finqa_parser.parse_record("train", raw)

    assert rec["record_id"] == "abc123"
    assert rec["filename"] == "sample_10k.txt"
    assert rec["split"] == "train"
    assert rec["question"] == "What is net income in 2020?"
    assert rec["answer"] == "100"
    assert "subtract" in rec["program_ops"]
    assert "difference" in rec["question_types"]
    assert rec["table_row_count"] == 2
    assert "table_structure" in rec
    assert "context_chunks" in rec
    assert "entities" in rec


def test_parse_record_handles_missing_program_with_edge_tag():
    raw = {
        "id": "xyz",
        "filename": "doc.txt",
        "pre_text": [],
        "post_text": [],
        "table": [["Metric", "2020"], ["Revenue", "-"]],
        "qa": {
            "question": "What is revenue?",
            "answer": "",
            "program": "",
        },
    }

    rec = finqa_parser.parse_record("private_test", raw)

    assert rec["program"] == ""
    assert "missing_program" in rec["edge_case_tags"]
    assert "missing_value_cells" in rec["edge_case_tags"]


def test_write_jsonl_round_trip(tmp_path: Path):
    rows = [
        {"record_id": "1", "split": "train"},
        {"record_id": "2", "split": "dev"},
    ]
    out = tmp_path / "records.jsonl"

    count = finqa_parser.write_jsonl(str(out), rows)

    assert count == 2
    loaded = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert loaded == rows


def test_extract_entities_populates_company_and_metrics():
    entities = extract_entities(
        question="What is revenue?",
        filename="AAPL/2020/page_1.json",
        pre_text=[],
        post_text=[],
        table=[["Metric", "2020"], ["Operating revenue", "100"]],
    )

    assert "AAPL" in entities["company_names"]
    assert "Operating revenue" in entities["metric_names"]


def test_per_share_rows_not_scaled_by_millions():
    result = normalize_table(
        [["Metric", "2020"], ["Operating revenue", "10"], ["Diluted EPS", "1.25"]],
        text_context="in millions, except per share data",
    )

    revenue, eps = result["rows"]
    assert revenue[1]["numeric_value_base"] == 10_000_000.0
    assert eps[1]["scale_multiplier"] == 1.0
    assert eps[1]["numeric_value_base"] == eps[1]["numeric_value"]


def test_multi_tiered_header_merging():
    result = normalize_table(
        [
            ["", "2020", "2019"],
            ["", "Three Months Ended", "Three Months Ended"],
            ["Revenue", "100", "90"],
        ]
    )

    assert result["header"][1] == "2020 Three Months Ended"
    assert len(result["rows"]) == 1
    assert result["row_labels"] == ["Revenue"]
