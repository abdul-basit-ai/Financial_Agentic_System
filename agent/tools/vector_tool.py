"""Semantic search tool over document narrative chunks using pgvector."""

from __future__ import annotations

import os
from typing import Any
import psycopg2
from psycopg2.extensions import connection
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from agent.tools.base import ToolResult

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class VectorSearchInput(BaseModel):
    query_text: str = Field(..., description="Semantic search query, e.g. 'reasons for cloud revenue growth'")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")
    record_id: str | None = Field(default=None, description="Filter search to a specific document record")
    split: str | None = Field(default=None, description="Filter search by dataset split ('train', 'dev', 'test')")


class VectorChunkRecord(BaseModel):
    record_id: str
    filename: str | None
    section: str
    chunk_index: int
    text_content: str
    similarity_score: float


class VectorSearchOutput(BaseModel):
    query: str
    chunks: list[VectorChunkRecord]
    total_found: int


class VectorRetrievalTool:
    """Manages dense semantic search over Postgres pgvector."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        dbname: str | None = None,
        user: str | None = None,
        password: str | None = None,
        model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        self.host = host or os.getenv("POSTGRES_HOST", "localhost")
        self.port = port or int(os.getenv("POSTGRES_PORT", "5432"))
        self.dbname = dbname or os.getenv("POSTGRES_DB", "financial_agent")
        self.user = user or os.getenv("POSTGRES_USER", "postgres")
        self.password = password or os.getenv("POSTGRES_PASSWORD", "password")

        self.model = SentenceTransformer(model_name)
        self._conn: connection | None = None

    def _get_connection(self) -> connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.dbname,
                user=self.user,
                password=self.password,
            )
        return self._conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
            self._conn = None

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        record_id: str | None = None,
        split: str | None = None,
    ) -> VectorSearchOutput:
        conn = self._get_connection()
        query_emb = self.model.encode(query_text, normalize_embeddings=True).tolist()

        sql = """
        SELECT
            record_id,
            filename,
            section,
            chunk_index,
            text_content,
            1.0 - (embedding <=> %(embedding)s::vector) AS similarity
        FROM document_chunks
        WHERE (%(record_id)s IS NULL OR record_id = %(record_id)s)
          AND (%(split)s IS NULL OR split = %(split)s)
        ORDER BY embedding <=> %(embedding)s::vector
        LIMIT %(limit)s;
        """

        # Pass str(query_emb) so psycopg2 formats '[...]' rather than PostgreSQL array '{...}'
        params: dict[str, Any] = {
            "embedding": str(query_emb),
            "record_id": record_id,
            "split": split,
            "limit": top_k,
        }

        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        records = [
            VectorChunkRecord(
                record_id=r[0],
                filename=r[1],
                section=r[2],
                chunk_index=r[3],
                text_content=r[4],
                similarity_score=float(r[5]),
            )
            for r in rows
        ]

        return VectorSearchOutput(
            query=query_text,
            chunks=records,
            total_found=len(records),
        )


def vector_retrieval_tool(
    payload: VectorSearchInput, client: VectorRetrievalTool | None = None
) -> ToolResult[VectorSearchOutput]:
    """Instrumented tool entrypoint for semantic vector retrieval."""
    instance = client or VectorRetrievalTool()
    return ToolResult.execute_instrumented(
        tool_name="vector_retrieval",
        fn=instance.search,
        query_text=payload.query_text,
        top_k=payload.top_k,
        record_id=payload.record_id,
        split=payload.split,
    )