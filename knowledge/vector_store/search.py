"""Exact, scoped search over the FinQA embedding store."""

from __future__ import annotations

import math
import os
from typing import Any

from psycopg2.extensions import connection

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

EXACT_SEARCH = """
WITH scoped AS MATERIALIZED (
    SELECT id, record_id, section, chunk_index, embedding
    FROM document_chunks AS chunk
    WHERE embedding IS NOT NULL
      AND (%(record_id)s IS NULL OR record_id = %(record_id)s)
      AND (%(record_ids)s::text[] IS NULL OR record_id = ANY(%(record_ids)s::text[]))
      AND (%(company)s IS NULL OR upper(split_part(record_id, '/', 1)) = %(company)s)
      AND (%(split)s IS NULL OR split = %(split)s)
      AND coalesce(to_jsonb(chunk)->>'embedding_model', %(default_model)s) = %(model)s
), ranked AS (
    SELECT id, record_id, section, chunk_index, embedding <=> %(embedding)s::vector AS distance
    FROM scoped
    ORDER BY distance, record_id, section, chunk_index
    LIMIT %(limit)s
)
SELECT chunk.record_id, chunk.filename, chunk.section, chunk.chunk_index, chunk.text_content,
       1.0 - ranked.distance AS similarity, chunk.sentence_start, chunk.sentence_end,
       to_jsonb(chunk)->>'provenance_version' AS provenance_version
FROM ranked JOIN document_chunks AS chunk USING (id)
ORDER BY ranked.distance, ranked.record_id, ranked.section, ranked.chunk_index
"""


def embedding_model_name(model_name: str | None = None) -> str:
    return model_name or os.getenv("EMBEDDING_MODEL") or EMBEDDING_MODEL_NAME


def embedding_values(embedding: Any) -> list[float]:
    values = [float(value) for value in embedding]
    if (
        len(values) != EMBEDDING_DIM
        or not all(math.isfinite(value) for value in values)
        or not any(values)
    ):
        raise ValueError(
            f"Expected a finite, nonzero {EMBEDDING_DIM}-dimension embedding"
        )
    return values


def search_embeddings(
    conn: connection,
    embedding: list[float],
    model: str,
    top_k: int,
    *,
    record_id: str | None = None,
    record_ids: list[str] | None = None,
    company: str | None = None,
    split: str | None = None,
) -> list[tuple[Any, ...]]:
    params = {
        "embedding": str(embedding),
        "model": model,
        "default_model": EMBEDDING_MODEL_NAME,
        "record_id": record_id,
        "record_ids": record_ids,
        "company": company,
        "split": split,
        "limit": top_k,
    }
    with conn.cursor() as cursor:
        cursor.execute(EXACT_SEARCH, params)
        return cursor.fetchall()
