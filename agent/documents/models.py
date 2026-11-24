from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    evidence_id: str
    document_id: str
    filename: str
    document_hash: str
    page: int = Field(ge=1)
    text: str
    bbox: list[float] = Field(default_factory=list)
    table_id: str | None = None
    row_index: int | None = None
    column_index: int | None = None
    row_label: str | None = None
    column_label: str | None = None
    raw_value: str | None = None
    numeric_value: float | None = None
    normalized_amount: float | None = None
    unit: str | None = None
    scale: float = 1.0
    currency: str | None = None
    quantity_kind: str = "number"
    units_ambiguous: bool = False
    period: str | None = None
    extraction_method: str = "native"
    extraction_confidence: float | None = None

    def citation(self) -> dict[str, Any]:
        return {**self.model_dump(exclude_none=True), "quote": self.text}
