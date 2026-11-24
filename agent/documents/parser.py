from __future__ import annotations

import hashlib
import math
import os
import re
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pymupdf
from numpy.typing import NDArray

from agent.documents.models import Evidence
from ingestion.normalizer import clean_text, normalize_cell
from ingestion.table_parser import parse_table_structure
from ingestion.unit_normalizer import currency_hints, parse_amount, scale_hints

_MAX_OCR_PIXELS = 12_000_000
_OCR_METHOD = "ocr"
_OCR_TABLE_METHOD = "ocr_verified_table"
_PERIOD = re.compile(r"\b(?:19|20)\d{2}\b")
_ALPHA = re.compile(r"[A-Za-z]")
_MISSING_LANGUAGE = "requires OCR. Configure English Tesseract language data and retry."

Box = tuple[float, float, float, float]
Block = tuple[Box, str]
Word = tuple[Box, str, int, int]


class DocumentParseError(ValueError):
    pass


class _OCRUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class _Line:
    lower: int
    upper: int
    start: int
    end: int

    @property
    def position(self) -> float:
        return (self.lower + self.upper) / 2


@dataclass(frozen=True)
class _Grid:
    columns: tuple[float, ...]
    rows: tuple[float, ...]

    @property
    def box(self) -> Box:
        return self.columns[0], self.rows[0], self.columns[-1], self.rows[-1]

    def cell(self, row: int, column: int) -> Box:
        return (
            self.columns[column],
            self.rows[row],
            self.columns[column + 1],
            self.rows[row + 1],
        )


@dataclass(frozen=True)
class _Raster:
    gray: NDArray[np.uint8]
    horizontal: list[_Line]
    vertical: list[_Line]
    grids: list[_Grid]


@dataclass(frozen=True)
class _Source:
    document_id: str
    filename: str
    digest: str
    page_number: int

    @property
    def prefix(self) -> str:
        return f"{self.document_id}:p{self.page_number}"

    def evidence(
        self, key: str, text: str, box: Box, method: str, **fields: Any
    ) -> Evidence:
        return Evidence(
            evidence_id=f"{self.prefix}:{key}",
            document_id=self.document_id,
            filename=self.filename,
            document_hash=self.digest,
            page=self.page_number,
            text=text,
            bbox=list(box),
            extraction_method=method,
            **fields,
        )


def _box(values: Any) -> Box:
    return float(values[0]), float(values[1]), float(values[2]), float(values[3])


def _page_bounds(page: pymupdf.Page, box: Box) -> Box:
    return _box(pymupdf.Rect(box) * page.derotation_matrix)


def _ink_lines(mask: NDArray[np.bool_], length: int, thickness: int) -> list[_Line]:
    lines: list[_Line] = []
    for offset in np.flatnonzero(mask.sum(axis=1) >= length):
        index = int(offset)
        transitions = np.flatnonzero(np.diff(mask[index], prepend=False, append=False))
        for left, right in transitions.reshape(-1, 2):
            start, end = int(left), int(right)
            if end - start < length:
                continue
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i]
                if (
                    index - line.upper <= 1
                    and abs(start - line.start) <= thickness
                    and abs(end - line.end) <= thickness
                ):
                    lines.pop(i)
                    lines.append(
                        _Line(
                            line.lower,
                            index,
                            min(start, line.start),
                            max(end, line.end),
                        )
                    )
                    break
            else:
                lines.append(_Line(index, index, start, end))
            if len(lines) > 512:
                raise DocumentParseError(
                    "Too many raster ruling lines to recover tables reliably."
                )
    return [line for line in lines if line.upper - line.lower < thickness]


def _grids(
    horizontal: list[_Line],
    vertical: list[_Line],
    page: pymupdf.Page,
    width: int,
    height: int,
    tolerance: float,
) -> list[_Grid]:
    by_horizontal: dict[int, list[tuple[bool, int]]] = defaultdict(list)
    by_vertical: dict[int, list[tuple[bool, int]]] = defaultdict(list)
    for h_index, h in enumerate(horizontal):
        for v_index, v in enumerate(vertical):
            if (
                h.start - tolerance <= v.position <= h.end + tolerance
                and v.start - tolerance <= h.position <= v.end + tolerance
            ):
                by_horizontal[h_index].append((False, v_index))
                by_vertical[v_index].append((True, h_index))

    found: list[_Grid] = []
    visited: set[tuple[bool, int]] = set()
    for h_index in by_horizontal:
        queue = [(True, h_index)]
        rows: set[int] = set()
        cols: set[int] = set()
        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            is_horizontal, index = current
            if is_horizontal:
                rows.add(index)
                queue.extend(by_horizontal[index])
            else:
                cols.add(index)
                queue.extend(by_vertical[index])
        if len(rows) < 3 or len(cols) < 2:
            continue
        if (
            len(cols) < 3
            or any(len(by_horizontal[index]) != len(cols) for index in rows)
            or (len(rows) - 1) * (len(cols) - 1) > 1800
        ):
            raise DocumentParseError(
                "A ruled table has merged, incomplete, or oversized cells."
            )
        x = tuple(
            sorted(vertical[index].position * page.rect.width / width for index in cols)
        )
        y = tuple(
            sorted(
                horizontal[index].position * page.rect.height / height for index in rows
            )
        )
        if any(b - a < 10 for a, b in zip(x, x[1:])) or any(
            b - a < 7 for a, b in zip(y, y[1:])
        ):
            raise DocumentParseError(
                "A ruled table has cells too small to recover reliably."
            )
        found.append(_Grid(x, y))
    return sorted(found, key=lambda grid: (grid.rows[0], grid.columns[0]))


def _rasterize(page: pymupdf.Page, dpi: int, threshold: int) -> _Raster:
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
    if pix.width * pix.height > _MAX_OCR_PIXELS:
        raise DocumentParseError("Raster dimensions exceed the supported OCR size.")
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width))
    dark = gray < threshold
    thickness = max(3, math.ceil(dpi * 0.055))
    horizontal = _ink_lines(dark, math.ceil(dpi * 0.50), thickness)
    vertical = _ink_lines(dark.T, math.ceil(dpi * 0.25), thickness)
    grids = _grids(horizontal, vertical, page, pix.width, pix.height, thickness)
    return _Raster(gray, horizontal, vertical, grids)


def _ocr_dpis(page: pymupdf.Page) -> tuple[int, int]:
    try:
        configured = int(os.getenv("DOCUMENT_OCR_DPI", "240"))
    except ValueError as exc:
        raise DocumentParseError("DOCUMENT_OCR_DPI must be an integer.") from exc
    if not 160 <= configured <= 360:
        raise DocumentParseError("DOCUMENT_OCR_DPI must be between 160 and 360.")
    area = page.rect.width * page.rect.height
    upper = min(360, math.floor(71 * math.sqrt(_MAX_OCR_PIXELS / area)))
    if upper < 176:
        raise DocumentParseError(
            "The scanned page is too large to verify at two OCR resolutions."
        )
    first = min(configured, upper - 16)
    return first, min(upper, first + 60)


def _words(page: pymupdf.Page, textpage: pymupdf.TextPage | None = None) -> list[Word]:
    return [
        (_box(word), str(word[4]), int(word[5]), int(word[6]))
        for word in page.get_text("words", textpage=textpage, sort=True)
    ]


def _grid_ocr(page: pymupdf.Page, raster: _Raster, dpi: int) -> list[Word]:
    cleaned = raster.gray.copy()
    for view, lines in ((cleaned, raster.horizontal), (cleaned.T, raster.vertical)):
        for line in lines:
            view[
                max(0, line.lower - 2) : line.upper + 3,
                max(0, line.start - 2) : line.end + 2,
            ] = 255
    height, width = cleaned.shape
    gray_pixmap = pymupdf.Pixmap(pymupdf.csGRAY, width, height, cleaned.tobytes(), 0)
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, gray_pixmap)
    pixmap.set_dpi(dpi, dpi)
    try:
        data = pixmap.pdfocr_tobytes(
            language="eng", tessdata=os.getenv("TESSDATA_PREFIX")
        )
    except Exception as exc:
        raise _OCRUnavailableError from exc
    with pymupdf.open(stream=data, filetype="pdf") as document:
        ocr_page = document[0]
        sx = page.rect.width / ocr_page.rect.width
        sy = page.rect.height / ocr_page.rect.height
        return [
            ((x0 * sx, y0 * sy, x1 * sx, y1 * sy), text, block, line)
            for (x0, y0, x1, y1), text, block, line in _words(ocr_page)
        ]


def _midpoint(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _inside(box: Box, x: float, y: float) -> bool:
    return box[0] <= x < box[2] and box[1] <= y < box[3]


def _word_blocks(words: list[Word], grids: list[_Grid]) -> list[Block]:
    by_block: dict[int, dict[int, list[tuple[Box, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for box, text, block, line in words:
        x, y = _midpoint(box)
        if not any(_inside(grid.box, x, y) for grid in grids):
            by_block[block][line].append((box, text))
    out: list[Block] = []
    for lines in by_block.values():
        entries = [word for line in lines.values() for word in line]
        bbox = (
            min(box[0] for box, _ in entries),
            min(box[1] for box, _ in entries),
            max(box[2] for box, _ in entries),
            max(box[3] for box, _ in entries),
        )
        text = "\n".join(" ".join(word for _, word in line) for line in lines.values())
        out.append((bbox, text))
    return out


def _grid_text(
    page: pymupdf.Page, raster: _Raster, grid: _Grid, words: list[Word]
) -> list[list[str]]:
    cells: dict[tuple[int, int], list[tuple[float, float, str]]] = defaultdict(list)
    for box, word, _, _ in words:
        x, y = _midpoint(box)
        if not _inside(grid.box, x, y):
            continue
        column = bisect_right(grid.columns, x) - 1
        row = bisect_right(grid.rows, y) - 1
        bounds = grid.cell(row, column)
        if any(
            bound - pos > 1.5 for bound, pos in zip(bounds[:2], box[:2], strict=True)
        ) or any(
            pos - bound > 1.5 for pos, bound in zip(box[2:], bounds[2:], strict=True)
        ):
            raise DocumentParseError(
                "Recognized text overlaps a scanned table cell boundary."
            )
        cells[row, column].append((box[1], box[0], word))
    matrix = [
        [
            clean_text(
                " ".join(
                    word
                    for _, _, word in sorted(
                        cells[row, column],
                        key=lambda item: (round(item[0] / 3), item[1]),
                    )
                )
            )
            for column in range(len(grid.columns) - 1)
        ]
        for row in range(len(grid.rows) - 1)
    ]
    height, width = raster.gray.shape
    sx = width / page.rect.width
    sy = height / page.rect.height
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            if value:
                continue
            x0, y0, x1, y1 = grid.cell(row, column)
            pixels = raster.gray[
                math.ceil((y0 + 2) * sy) : math.floor((y1 - 2) * sy),
                math.ceil((x0 + 2) * sx) : math.floor((x1 - 2) * sx),
            ]
            if np.count_nonzero(pixels < 155) > max(12, pixels.size * 0.004):
                raise DocumentParseError(
                    "A scanned table cell contains unreadable content."
                )
    return matrix


def _unverified_table(words: list[Word]) -> bool:
    lines: list[list[tuple[Box, str]]] = []
    for box, text, _, _ in sorted(words, key=lambda word: _midpoint(word[0])[1]):
        y = _midpoint(box)[1]
        if not lines or y - _midpoint(lines[-1][0][0])[1] > 4:
            lines.append([])
        lines[-1].append((box, text))
    columns: list[tuple[Box, Box]] = []
    for row in lines:
        row.sort(key=lambda item: item[0][0])
        numbers = [box for box, text in row if parse_amount(text)[0] is not None]
        if len(numbers) >= 2 and any(
            _ALPHA.search(text) and box[2] < numbers[0][0] for box, text in row
        ):
            first, *_, last = numbers
            if last[0] - first[2] >= 12:
                columns.append((first, last))

    def aligned(x: Box, y: Box) -> bool:
        return min(abs(x[0] - y[0]), abs(x[2] - y[2])) < 8

    return any(
        sum(aligned(first, x) and aligned(last, y) for first, last in columns) >= 3
        for x, y in columns
    )


def _unit_context(blocks: list[Block], table_box: Box) -> str:
    left, top, right, _ = table_box
    captions = []
    for (x0, _, x1, bottom), content in blocks:
        distance = top - bottom
        if distance < -1 or distance >= 160 or x1 < left or x0 > right:
            continue
        for line in content.splitlines():
            if scale_hints(line, declaration_only=True) or currency_hints(line):
                captions.append((distance, line.strip()))
    return " ".join(line for _, line in sorted(captions)[:3])[:500]


def _header_columns(matrix: list[list[Any]], depth: int) -> list[str]:
    return [
        " ".join(
            clean_text(matrix[row][column])
            for row in range(depth)
            if column < len(matrix[row])
        ).strip()
        for column in range(max(map(len, matrix)))
    ]


def _table_cells(
    source: _Source,
    index: int,
    matrix: list[list[Any]],
    cell_box: Callable[[int, int], Box],
    context: str,
    method: str,
) -> list[Evidence]:
    depth = int(parse_table_structure(matrix)["data_start_row"])
    columns = _header_columns(matrix, depth)
    strict = method == _OCR_TABLE_METHOD
    if strict and (
        len(columns) < 2
        or any(
            not (_ALPHA.search(column) or _PERIOD.fullmatch(column))
            for column in columns[1:]
        )
        or len({column.casefold() for column in columns[1:]}) != len(columns) - 1
    ):
        raise DocumentParseError(
            "Scanned table column headings are missing or ambiguous."
        )
    items = []
    labels = set()
    for row_index, row in enumerate(matrix[depth:], depth):
        label = clean_text(row[0]) if row else ""
        if strict and any(clean_text(cell) for cell in row[1:]):
            if (
                not (_ALPHA.search(label) or _PERIOD.fullmatch(label))
                or label.casefold() in labels
            ):
                raise DocumentParseError(
                    "Scanned table row headings are missing or ambiguous."
                )
            labels.add(label.casefold())
        for column_index, raw in enumerate(row[1:], 1):
            column = columns[column_index]
            cell = normalize_cell(raw, context, row_label=label, column_header=column)
            if cell["numeric_value"] is None:
                if strict and not cell["is_missing"]:
                    raise DocumentParseError(
                        "A scanned table cell does not contain a verified financial amount."
                    )
                continue
            if strict and cell["units_ambiguous"]:
                raise DocumentParseError("Scanned table units conflict.")
            raw_text = str(raw)
            text = f"{label} | {column} | {raw_text}" + (
                f" ({context})" if context else ""
            )
            items.append(
                source.evidence(
                    f"t{index}:r{row_index}:c{column_index}",
                    text,
                    cell_box(row_index, column_index),
                    method,
                    table_id=f"p{source.page_number}:t{index}",
                    row_index=row_index,
                    column_index=column_index,
                    row_label=label,
                    column_label=column,
                    raw_value=raw_text,
                    numeric_value=cell["numeric_value"],
                    normalized_amount=cell["numeric_value_base"],
                    unit=cell["unit_label"],
                    scale=float(cell["scale_multiplier"] or 1),
                    currency=cell["currency"],
                    period=column or None,
                    quantity_kind=cell["quantity_kind"],
                    units_ambiguous=cell["units_ambiguous"],
                )
            )
    if strict and not items:
        raise DocumentParseError("The scanned table has no verified numeric cells.")
    return items


def _native_blocks(
    page: pymupdf.Page, textpage: pymupdf.TextPage | None = None
) -> list[Block]:
    return [
        (_box(block), str(block[4]).strip())
        for block in page.get_text("blocks", textpage=textpage, sort=True)
        if len(block) >= 7 and block[6] == 0 and str(block[4]).strip()
    ]


def _scan_page(
    page: pymupdf.Page, source: _Source
) -> tuple[list[Block], list[Evidence]]:
    first_dpi, second_dpi = _ocr_dpis(page)
    first = _rasterize(page, first_dpi, 165)
    if int(first.gray.min()) >= 245:
        return [], []
    if not first.grids:
        try:
            textpage = page.get_textpage_ocr(
                language="eng",
                dpi=first_dpi,
                full=True,
                tessdata=os.getenv("TESSDATA_PREFIX"),
            )
        except Exception as exc:
            raise _OCRUnavailableError from exc
        words = _words(page, textpage)
        if not words:
            raise DocumentParseError(
                "No readable text could be recovered from the scanned page."
            )
        if _unverified_table(words):
            raise DocumentParseError(
                "A scanned table without a verifiable grid could not be recovered safely."
            )
        return _native_blocks(page, textpage), []

    second = _rasterize(page, second_dpi, 185)
    if len(first.grids) != len(second.grids):
        raise DocumentParseError("Scanned table layout differs across OCR passes.")
    first_words = _grid_ocr(page, first, first_dpi)
    second_words = _grid_ocr(page, second, second_dpi)
    first_blocks = _word_blocks(first_words, first.grids)
    second_blocks = _word_blocks(second_words, second.grids)
    items = []
    for index, (grid, check) in enumerate(zip(first.grids, second.grids, strict=True)):
        if (
            len(grid.columns) != len(check.columns)
            or len(grid.rows) != len(check.rows)
            or any(
                abs(a - b) > 1.5
                for a, b in zip(
                    (*grid.columns, *grid.rows),
                    (*check.columns, *check.rows),
                    strict=True,
                )
            )
        ):
            raise DocumentParseError(
                "Scanned table cell boundaries differ across OCR passes."
            )
        matrix = _grid_text(page, first, grid, first_words)
        context = _unit_context(first_blocks, grid.box)
        second_context = _unit_context(second_blocks, check.box)
        if (
            matrix != _grid_text(page, second, check, second_words)
            or context.casefold() != second_context.casefold()
        ):
            raise DocumentParseError(
                "Scanned table text or units differ across OCR passes."
            )
        items.extend(
            _table_cells(
                source,
                index,
                matrix,
                lambda row, column: _page_bounds(page, grid.cell(row, column)),
                context,
                _OCR_TABLE_METHOD,
            )
        )
    return [(_page_bounds(page, box), text) for box, text in first_blocks], items


def _looks_scanned(page: pymupdf.Page) -> bool:
    image_bounds = [
        pymupdf.Rect(item["bbox"]) * page.rotation_matrix
        for item in page.get_image_info()
    ]
    return bool(image_bounds) and (
        len(page.get_text("text").strip()) < 20
        or any(
            (page.rect & box).get_area() >= page.rect.get_area() * 0.65
            for box in image_bounds
        )
    )


def _parse_page(page: pymupdf.Page, source: _Source) -> list[Evidence]:
    if page.rect.width * page.rect.height > 12_000_000:
        raise DocumentParseError(
            f"Page {source.page_number} exceeds supported dimensions."
        )
    if _looks_scanned(page):
        try:
            blocks, cells = _scan_page(page, source)
        except DocumentParseError as exc:
            raise DocumentParseError(
                f"Page {source.page_number} could not be fully extracted. {exc}"
            ) from exc
        except _OCRUnavailableError as exc:
            raise DocumentParseError(
                f"Page {source.page_number} {_MISSING_LANGUAGE}"
            ) from exc
        method = _OCR_METHOD
    else:
        blocks = _native_blocks(page)
        cells = []
        method = "native"
        try:
            for index, table in enumerate(page.find_tables().tables):
                matrix = table.extract()
                if len(matrix) < 2:
                    continue
                table_bbox = _box(table.bbox)

                def cell_box(row: int, column: int) -> Box:
                    cells_in_row = table.rows[row].cells
                    return _box(cells_in_row[column] or table_bbox)

                cells.extend(
                    _table_cells(
                        source,
                        index,
                        matrix,
                        cell_box,
                        _unit_context(blocks, table_bbox),
                        method,
                    )
                )
        except Exception as exc:
            raise DocumentParseError(
                f"Page {source.page_number} table layout could not be extracted."
            ) from exc
    narrative = [
        source.evidence(
            f"b{index}:{part // 12000}", text[part : part + 12000], box, method
        )
        for index, (box, text) in enumerate(blocks)
        for part in range(0, len(text), 12000)
    ]
    return narrative + cells


def parse_pdf(
    content: bytes, document_id: str, filename: str
) -> tuple[list[Evidence], int]:
    if not content.startswith(b"%PDF-"):
        raise DocumentParseError("The uploaded file is not a PDF.")
    digest = hashlib.sha256(content).hexdigest()
    evidence: list[Evidence] = []
    try:
        document = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise DocumentParseError("The PDF could not be opened.") from exc
    with document:
        if document.is_encrypted:
            raise DocumentParseError("Remove the PDF password before uploading.")
        page_count = len(document)
        if page_count < 1 or page_count > int(os.getenv("MAX_DOCUMENT_PAGES", "1000")):
            raise DocumentParseError("The PDF exceeds the supported page count.")
        for page_index in range(page_count):
            page: pymupdf.Page = document.load_page(page_index)
            evidence.extend(
                _parse_page(
                    page, _Source(document_id, filename, digest, page_index + 1)
                )
            )
            if len(evidence) > 50000:
                raise DocumentParseError(
                    "The report exceeds the supported evidence limit."
                )
    if not evidence:
        raise DocumentParseError("No readable evidence was found in this report.")
    return evidence, page_count


def evidence_payload(items: list[Evidence]) -> list[dict[str, Any]]:
    return [item.model_dump(exclude_none=True) for item in items]
