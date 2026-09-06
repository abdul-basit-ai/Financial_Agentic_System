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
from sentence_transformers import SentenceTransformer

SPLITS = ["train", "dev", "test", "private_test"]
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

DDL_SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    record_id TEXT NOT NULL,
    filename TEXT,
    split TEXT,
    section TEXT NOT NULL,
    chunk_index INT NOT NULL,
    text_content TEXT NOT NULL,
    embedding vector({EMBEDDING_DIM})
);

CREATE UNIQUE INDEX IF NOT EXISTS chunk_unique_idx 
ON document_chunks (record_id, section, chunk_index);

CREATE INDEX IF NOT EXISTS chunk_record_idx 
ON document_chunks (record_id);

CREATE INDEX IF NOT EXISTS chunk_vector_idx 
ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
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
        model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        self.conn_params = {
            "host": host,
            "port": port,
            "dbname": dbname,
            "user": user,
            "password": password,
        }
        self.model = SentenceTransformer(model_name)
        self._conn = None

    def connect(self) -> None:
        self._conn = psycopg2.connect(**self.conn_params)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()

    def create_schema(self) -> None:
        assert self._conn is not None
        with self._conn.cursor() as cur:
            cur.execute(DDL_SCHEMA)
        self._conn.commit()

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
            section = str(ch.get("source", "context"))
            rows.append((record_id, filename, split, section, idx, text))
        return rows

    def insert_batch(self, rows_batch: list[tuple[Any, ...]]) -> int:
        if not rows_batch:
            return 0
        assert self._conn is not None

        texts = [r[5] for r in rows_batch]
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        payload = [
            (r[0], r[1], r[2], r[3], r[4], r[5], embeddings[i].tolist())
            for i, r in enumerate(rows_batch)
        ]

        query = """
        INSERT INTO document_chunks (
            record_id, filename, split, section, chunk_index, text_content, embedding
        ) VALUES %s
        ON CONFLICT (record_id, section, chunk_index)
        DO UPDATE SET
            text_content = EXCLUDED.text_content,
            embedding = EXCLUDED.embedding;
        """

        with self._conn.cursor() as cur:
            execute_values(cur, query, payload)
        self._conn.commit()
        return len(payload)

    def query_similarity(self, text: str, top_k: int = 5) -> list[dict[str, Any]]:
        assert self._conn is not None
        query_emb = self.model.encode(text, normalize_embeddings=True).tolist()

        sql = """
        SELECT record_id, section, chunk_index, text_content, 1 - (embedding <=> %s::vector) AS similarity
        FROM document_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (query_emb, query_emb, top_k))
            results = cur.fetchall()

        return [
            {
                "record_id": r[0],
                "section": r[1],
                "chunk_index": r[2],
                "text_content": r[3],
                "similarity": float(r[4]),
            }
            for r in results
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load FinQA narrative chunks into pgvector")
    parser.add_argument("--normalized-dir", default="data/processed/normalized")
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--host", default=os.getenv("POSTGRES_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("POSTGRES_PORT", "5432")))
    parser.add_argument("--dbname", default=os.getenv("POSTGRES_DB", "financial_agent"))
    parser.add_argument("--user", default=os.getenv("POSTGRES_USER", "postgres"))
    parser.add_argument("--password", default=os.getenv("POSTGRES_PASSWORD", "password"))
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

        for split in args.splits:
            path = os.path.join(args.normalized_dir, f"finqa_{split}_normalized.jsonl")
            if not os.path.exists(path):
                continue
            for rec in read_jsonl(path):
                rows = loader.load_chunks_from_record(rec)
                current_batch.extend(rows)
                if len(current_batch) >= args.batch_size:
                    inserted = loader.insert_batch(current_batch)
                    total_chunks += inserted
                    current_batch = []
                    print(f"Indexed {total_chunks} chunks...")

        if current_batch:
            inserted = loader.insert_batch(current_batch)
            total_chunks += inserted

        print(f"Vector loading complete: total_chunks={total_chunks}")
    finally:
        loader.close()


if __name__ == "__main__":
    main()