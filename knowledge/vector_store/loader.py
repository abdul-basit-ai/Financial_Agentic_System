"""PostgreSQL pgvector chunk loader for FinQA narrative context.

Embeds and loads context.chunks into pgvector for semantic retrieval.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from typing import Any

import psycopg2
from psycopg2.extras import execute_values

from knowledge.vector_store.search import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    embedding_model_name,
    embedding_values,
    search_embeddings,
)

SPLITS = ["train", "dev", "test", "private_test"]

DDL_SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    record_id TEXT NOT NULL,
    filename TEXT,
    split TEXT,
    section TEXT NOT NULL,
    chunk_index INT NOT NULL,
    sentence_start INT,
    sentence_end INT,
    text_content TEXT NOT NULL,
    embedding vector({EMBEDDING_DIM})
);

ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS sentence_start INT;
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS sentence_end INT;
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS provenance_version INT;
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding_model TEXT
    NOT NULL DEFAULT '{EMBEDDING_MODEL_NAME}';

CREATE UNIQUE INDEX IF NOT EXISTS chunk_unique_idx
ON document_chunks (record_id, section, chunk_index);

CREATE INDEX IF NOT EXISTS chunk_record_idx
ON document_chunks (record_id);

"""


def read_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


class VectorStoreLoader:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "financial_agent",
        user: str = "postgres",
        password: str = "password",
        model_name: str | None = None,
    ) -> None:
        self.conn_params: dict[str, Any] = {
            "host": host,
            "port": port,
            "dbname": dbname,
            "user": user,
            "password": password,
            "connect_timeout": 3,
        }
        from sentence_transformers import SentenceTransformer

        self.model_name = embedding_model_name(model_name)
        self.model = SentenceTransformer(self.model_name)
        self._conn = None

    def connect(self) -> None:
        self._conn = psycopg2.connect(**self.conn_params)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()

    def create_schema(self) -> None:
        assert self._conn is not None
        with self._conn, self._conn.cursor() as cur:
            cur.execute(DDL_SCHEMA)

    def ensure_vector_index(self, min_rows: int = 1000, lists: int = 100) -> bool:
        """Creates the ivfflat index once the table is large enough to populate it.

        Returns True if the index exists. Below min_rows the index would have
        empty clusters and approximate search can silently miss rows, so exact
        sequential scan is used instead (correct, just slower at small scale).
        """
        assert self._conn is not None
        with self._conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM document_chunks")
            count = int(cur.fetchone()[0])
            if count < min_rows:
                cur.execute("DROP INDEX IF EXISTS chunk_vector_idx")
                self._conn.commit()
                return False
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS chunk_vector_idx
                ON document_chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {int(lists)});
                """)
            self._conn.commit()
            return True

    def load_chunks_from_record(self, rec: dict[str, Any]) -> list[tuple[Any, ...]]:
        record_id = str(rec.get("record_id", ""))
        doc = rec.get("document", {}) if isinstance(rec.get("document"), dict) else {}
        filename = str(doc.get("filename", ""))
        split = str(rec.get("split", ""))

        ctx = rec.get("context", {}) if isinstance(rec.get("context"), dict) else {}
        raw_chunks = ctx.get("chunks", [])

        rows = []
        for idx, ch in enumerate(raw_chunks):
            if not isinstance(ch, dict):
                continue
            text = str(ch.get("text", "")).strip()
            if not text:
                continue

            if ch.get("low_quality"):
                continue
            section = str(ch.get("source", "context"))
            sentence_start = ch.get("start_sentence")
            sentence_end = ch.get("end_sentence")
            rows.append(
                (
                    record_id,
                    filename,
                    split,
                    section,
                    idx,
                    sentence_start if isinstance(sentence_start, int) else None,
                    sentence_end if isinstance(sentence_end, int) else None,
                    text,
                    2 if ctx.get("source_indices_explicit") else None,
                )
            )
        return rows

    def insert_batch(
        self,
        rows_batch: list[tuple[Any, ...]],
        *,
        replace_record_ids: list[str] | None = None,
    ) -> int:
        if not rows_batch and not replace_record_ids:
            return 0
        assert self._conn is not None

        texts = [r[7] for r in rows_batch]
        embeddings = (
            self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            if texts
            else []
        )

        payload = [
            (*row, self.model_name, embedding_values(embedding))
            for row, embedding in zip(rows_batch, embeddings, strict=True)
        ]

        query = """
        INSERT INTO document_chunks (
            record_id, filename, split, section, chunk_index,
            sentence_start, sentence_end, text_content, provenance_version, embedding_model, embedding
        ) VALUES %s
        ON CONFLICT (record_id, section, chunk_index)
        DO UPDATE SET
            filename = EXCLUDED.filename,
            split = EXCLUDED.split,
            sentence_start = EXCLUDED.sentence_start,
            sentence_end = EXCLUDED.sentence_end,
            text_content = EXCLUDED.text_content,
            provenance_version = EXCLUDED.provenance_version,
            embedding_model = EXCLUDED.embedding_model,
            embedding = EXCLUDED.embedding;
        """

        with self._conn, self._conn.cursor() as cur:
            if replace_record_ids:
                record_ids = sorted(set(replace_record_ids))
                cur.execute(
                    """
                    SELECT pg_advisory_xact_lock(hashtextextended(id, 0))
                    FROM unnest(%s::text[]) AS ids(id)
                    ORDER BY id
                    """,
                    (record_ids,),
                )
                cur.execute(
                    "DELETE FROM document_chunks WHERE record_id = ANY(%s)",
                    (record_ids,),
                )
            if payload:
                execute_values(cur, query, payload)
        return len(payload)

    def query_similarity(
        self, text: str, top_k: int = 5, record_id: str | None = None
    ) -> list[dict[str, Any]]:
        assert self._conn is not None
        query_emb = embedding_values(self.model.encode(text, normalize_embeddings=True))
        with self._conn:
            results = search_embeddings(
                self._conn, query_emb, self.model_name, top_k, record_id=record_id
            )

        return [
            {
                "record_id": r[0],
                "section": r[2],
                "chunk_index": r[3],
                "text_content": r[4],
                "similarity": float(r[5]),
            }
            for r in results
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load FinQA narrative chunks into pgvector"
    )
    parser.add_argument("--normalized-dir", default="data/processed/normalized")
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--host", default=os.getenv("POSTGRES_HOST", "localhost"))
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("POSTGRES_PORT", "5432"))
    )
    parser.add_argument("--dbname", default=os.getenv("POSTGRES_DB", "financial_agent"))
    parser.add_argument("--user", default=os.getenv("POSTGRES_USER", "postgres"))
    parser.add_argument(
        "--password", default=os.getenv("POSTGRES_PASSWORD", "password")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = VectorStoreLoader(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    )
    loader.connect()
    try:
        loader.create_schema()
        total_chunks = 0
        current_batch: list[tuple[Any, ...]] = []
        current_records: list[str] = []

        for split in args.splits:
            path = os.path.join(args.normalized_dir, f"finqa_{split}_normalized.jsonl")
            if not os.path.exists(path):
                continue
            for rec in read_jsonl(path):
                rows = loader.load_chunks_from_record(rec)
                current_records.append(rec["record_id"])
                current_batch.extend(rows)
                if max(len(current_batch), len(current_records)) >= args.batch_size:
                    inserted = loader.insert_batch(
                        current_batch, replace_record_ids=current_records
                    )
                    total_chunks += inserted
                    current_batch = []
                    current_records = []
                    print(f"Indexed {total_chunks} chunks...")

        if current_records:
            inserted = loader.insert_batch(
                current_batch, replace_record_ids=current_records
            )
            total_chunks += inserted

        print(f"Vector loading complete: total_chunks={total_chunks}, search=exact")
    finally:
        loader.close()


if __name__ == "__main__":
    main()
