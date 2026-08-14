import json
from pathlib import Path

from ingestion import parser as finqa_parser


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
