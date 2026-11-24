from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path
from unittest.mock import MagicMock

import pymupdf
import pytest

from agent.documents import parser
from agent.documents.parser import DocumentParseError, evidence_payload, parse_pdf
from agent.documents.service import DocumentService
from agent.documents.storage import FileStorage
from agent.server.store import RecordStore


def _native_statement(*, ruled: bool = True) -> bytes:
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text(
            (50, 50), "Consolidated statements in USD millions except per share data"
        )
        table = [
            ["Metric", "2021", "2020"],
            ["Operating income", "$1,200", "$900"],
            ["Debt in billions", "$3", "$2"],
            ["Diluted EPS", "$(1.25)", "$0.75"],
        ]
        columns = [50, 200, 310, 420]
        top = 80
        row_height = 24
        bottom = top + row_height * len(table)
        if ruled:
            for x in columns:
                page.draw_line((x, top), (x, bottom))
            for index in range(len(table) + 1):
                y = top + index * row_height
                page.draw_line((columns[0], y), (columns[-1], y))
        for index, row in enumerate(table):
            for x, text in zip(columns, row):
                page.insert_text((x + 6, top + row_height * index + 16), text)
        trailing_page = document.new_page()
        trailing_page.insert_text(
            (45, 65), "The annual report was published on February 15, 2022."
        )
        return document.tobytes()


def _scan_with_short_text() -> bytes:
    with pymupdf.open() as document:
        page = document.new_page(width=180, height=160)
        raster = page.get_pixmap()
        page.insert_image(page.rect, pixmap=raster)
        page.insert_text((20, 40), "Revenue was 18.")
        return document.tobytes()


def _rasterized_statement(scanned_pages: set[int], *, ruled: bool = True) -> bytes:
    with (
        pymupdf.open(stream=_native_statement(ruled=ruled), filetype="pdf") as source,
        pymupdf.open() as document,
    ):
        for index, original in enumerate(source):
            if index in scanned_pages:
                page = document.new_page(
                    width=original.rect.width, height=original.rect.height
                )
                page.insert_image(page.rect, pixmap=original.get_pixmap(dpi=160))
            else:
                document.insert_pdf(source, from_page=index, to_page=index)
        return document.tobytes()


@pytest.fixture
def english_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    configured = os.getenv("TESSDATA_PREFIX")
    paths = [
        Path(__file__).resolve().parent.parent / "tmp" / "tessdata",
        Path("/usr/share/tesseract-ocr/5/tessdata"),
    ]
    if configured:
        paths.insert(0, Path(configured))
    for path in paths:
        if (path / "eng.traineddata").is_file():
            monkeypatch.setenv("TESSDATA_PREFIX", str(path.resolve()))
            return
    pytest.skip("English Tesseract language data is unavailable")


def test_parse_native_statement_preserves_page_and_cell_sources() -> None:
    pdf = _native_statement()
    items, page_count = parse_pdf(pdf, "doc_fixture", "statement.pdf")
    assert page_count == 2
    assert {item.page for item in items} == {1, 2}
    assert {item.document_hash for item in items} == {sha256(pdf).hexdigest()}
    assert {item.document_id for item in items} == {"doc_fixture"}
    assert {item.extraction_method for item in items} == {"native"}
    assert len({item.evidence_id for item in items}) == len(items)
    assert [item.evidence_id for item in items] == [
        item.evidence_id for item in parse_pdf(pdf, "doc_fixture", "statement.pdf")[0]
    ]

    cells = {
        (item.row_label, item.column_label): item for item in items if item.table_id
    }
    income = cells[("Operating income", "2021")]
    assert (income.row_index, income.column_index) == (1, 1)
    assert income.evidence_id == "doc_fixture:p1:t0:r1:c1"
    assert income.raw_value == "$1,200"
    assert income.numeric_value == 1200
    assert income.normalized_amount == 1_200_000_000
    assert income.currency == "USD"
    assert income.bbox[0] >= 200
    assert income.bbox[1] >= 104
    assert income.bbox[2] <= 310
    assert income.bbox[3] <= 129
    assert cells[("Debt in billions", "2021")].normalized_amount == 3_000_000_000
    eps = cells[("Diluted EPS", "2021")]
    assert eps.numeric_value == -1.25
    assert eps.scale == 1
    assert any(
        "published on February 15" in item.text for item in items if item.page == 2
    )
    assert {row["evidence_id"] for row in evidence_payload(items)} == {
        item.evidence_id for item in items
    }


def test_ocr_failure_reports_page_and_successful_retry_marks_ocr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf = _scan_with_short_text()
    monkeypatch.setenv("DATABASE_URL", "")
    storage = MagicMock(spec=FileStorage)
    storage.put.return_value = "owner/scan.pdf"
    storage.read.return_value = pdf
    store = RecordStore(path=str(tmp_path / "workspace.sqlite3"))
    service = DocumentService(store, storage)
    queued = service.create("owner", "scan.pdf", pdf)
    document_id = queued["document_id"]
    assert queued["status"] == "queued"

    def missing_language(page: pymupdf.Page, **kwargs: object) -> pymupdf.TextPage:
        raise RuntimeError("English data unavailable")

    monkeypatch.setattr(pymupdf.Page, "get_textpage_ocr", missing_language)
    failed = service.ingest("owner", document_id)
    assert failed["status"] == "failed"
    assert "Page 1 requires OCR" in failed["error"]
    assert store.list("evidence", "owner") == []
    with pytest.raises(ValueError, match="ready"):
        service.evidence("owner", [document_id])

    def recognized_text(page: pymupdf.Page, **kwargs: object) -> pymupdf.TextPage:
        assert kwargs["language"] == "eng"
        return page.get_textpage()

    monkeypatch.setattr(pymupdf.Page, "get_textpage_ocr", recognized_text)
    ready = service.ingest("owner", document_id)
    assert ready["status"] == "ready"
    assert ready["page_count"] == 1
    assert ready["error"] is None
    items = service.evidence("owner", [document_id])
    assert {item.extraction_method for item in items} == {"ocr"}
    assert any(item.text == "Revenue was 18." for item in items)
    assert all(item.page == 1 for item in items)


def test_pdf_parser_rejects_invalid_encrypted_and_blank_inputs() -> None:
    with pytest.raises(DocumentParseError, match="not a PDF"):
        parse_pdf(b"not pdf", "invalid", "invalid.pdf")
    with pytest.raises(DocumentParseError, match="could not be opened"):
        parse_pdf(b"%PDF-1.7\ninvalid", "malformed", "malformed.pdf")
    with pymupdf.open() as document:
        document.new_page()
        blank = document.tobytes()
        encrypted = document.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="synthetic-owner",
            user_pw="synthetic-reader",
        )
    with pytest.raises(DocumentParseError, match="password"):
        parse_pdf(encrypted, "encrypted", "encrypted.pdf")
    with pytest.raises(DocumentParseError, match="No readable evidence"):
        parse_pdf(blank, "blank", "blank.pdf")


@pytest.mark.parametrize("scanned_pages", [{0}, {0, 1}], ids=["mixed", "fully_scanned"])
def test_real_ocr_recovers_ruled_table_and_narrative(
    english_ocr: None,
    scanned_pages: set[int],
) -> None:
    pdf = _rasterized_statement(scanned_pages)
    evidence, pages = parse_pdf(pdf, "scanned", "annual-report.pdf")
    assert pages == 2
    cells = {
        (item.row_label, item.column_label): item for item in evidence if item.table_id
    }
    assert len(cells) == 6
    assert {item.extraction_method for item in cells.values()} == {"ocr_verified_table"}
    assert {item.document_hash for item in evidence} == {sha256(pdf).hexdigest()}
    income = cells["Operating income", "2021"]
    assert income.raw_value == "$1,200"
    assert income.normalized_amount == 1_200_000_000
    assert income.period == "2021"
    assert income.currency == "USD"
    assert income.evidence_id == "scanned:p1:t0:r1:c1"
    assert income.bbox == pytest.approx([200, 104, 310, 128], abs=1)
    assert cells["Debt in billions", "2021"].normalized_amount == 3_000_000_000
    assert cells["Diluted EPS", "2021"].numeric_value == -1.25
    second_page = [item for item in evidence if item.page == 2]
    method = "ocr" if 1 in scanned_pages else "native"
    assert {item.extraction_method for item in second_page} == {method}
    assert any("published on February 15" in item.text for item in second_page)


def test_scanned_layout_does_not_accept_disagreeing_ocr_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = _rasterized_statement({0})
    passes = []

    def mismatched_ocr(
        page: pymupdf.Page, raster: parser._Raster, dpi: int
    ) -> list[parser.Word]:
        passes.append(dpi)
        matrix = [
            ["Metric", "2021", "2020"],
            ["Operating income", "$1,200" if len(passes) == 1 else "$1,700", "$900"],
            ["Debt in billions", "$3", "$2"],
            ["Diluted EPS", "$(1.25)", "$0.75"],
        ]
        (grid,) = raster.grids
        return [
            ((x + 3, y + 3, x + 20, y + 9), value, row, row)
            for row, cells in enumerate(matrix)
            for column, value in enumerate(cells)
            for x, y in [(grid.columns[column], grid.rows[row])]
        ]

    monkeypatch.setattr(parser, "_grid_ocr", mismatched_ocr)
    with pytest.raises(DocumentParseError, match="Page 1 .*text or units differ"):
        parse_pdf(pdf, "conflicting", "conflicting.pdf")
    assert len(set(passes)) == 2


def test_mixed_pdf_fails_with_page_number_when_ocr_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = _rasterized_statement({1})

    def unavailable(page: pymupdf.Page, **kwargs: object) -> pymupdf.TextPage:
        raise RuntimeError("English model is unavailable")

    monkeypatch.setattr(pymupdf.Page, "get_textpage_ocr", unavailable)
    with pytest.raises(DocumentParseError, match="Page 2 requires OCR"):
        parse_pdf(pdf, "mixed", "mixed.pdf")


def test_real_ocr_reports_unruled_financial_table_as_unsupported(
    english_ocr: None,
) -> None:
    pdf = _rasterized_statement({0}, ruled=False)
    with pytest.raises(DocumentParseError, match="Page 1 .*without a verifiable grid"):
        parse_pdf(pdf, "unruled", "unruled.pdf")


def test_real_engine_classifies_missing_english_data(
    english_ocr: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path))
    with pytest.raises(DocumentParseError, match="Page 1 requires OCR"):
        parse_pdf(_rasterized_statement({0}), "missing_language", "missing.pdf")
