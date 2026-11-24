"""Semantic search tool over document narrative chunks using pgvector."""

from __future__ import annotations

import os
from contextlib import closing
from threading import Lock

import psycopg2
from psycopg2.extensions import connection
from pydantic import BaseModel, Field

from agent.tools.base import ToolResult
from ingestion.graph_loader import source_page
from knowledge.vector_store.search import (
    embedding_model_name,
    embedding_values,
    search_embeddings,
)


class VectorSearchInput(BaseModel):
    query_text: str = Field(
        ...,
        description="Semantic search query, e.g. 'reasons for cloud revenue growth'",
    )
    top_k: int = Field(
        default=5, ge=1, le=20, description="Number of context chunks to retrieve"
    )
    record_id: str | None = Field(
        default=None, description="Filter search to a specific document record"
    )
    record_ids: list[str] | None = Field(
        default=None,
        description="Filter search to a set of document records (e.g. all filings of one company) — "
        "used by record anchoring so the top chunk identifies the right filing.",
    )
    company_identifier: str | None = Field(
        default=None,
        description="Scope search to one company's filings (record_id prefix, e.g. 'AMZN/'). "
        "Prevents cross-company narrative pollution when the planner names a company.",
    )
    split: str | None = Field(
        default=None,
        description="Filter search by dataset split ('train', 'dev', 'test')",
    )


class VectorChunkRecord(BaseModel):
    record_id: str
    filename: str | None
    section: str
    chunk_index: int
    text_content: str
    similarity_score: float
    evidence_id: str | None = None
    page: int | None = None
    sentence_start: int | None = None
    sentence_end: int | None = None
    gold_ids: list[str] = Field(default_factory=list)


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
        model_name: str | None = None,
    ) -> None:
        self.host = host or os.getenv("POSTGRES_HOST", "localhost")
        self.port = port or int(os.getenv("POSTGRES_PORT", "5432"))
        self.dbname = dbname or os.getenv("POSTGRES_DB", "financial_agent")
        self.user = user or os.getenv("POSTGRES_USER", "postgres")
        self.password = password or os.getenv("POSTGRES_PASSWORD", "password")

        self.model_name = embedding_model_name(model_name)
        self._encoder_lock = Lock()

    def _get_connection(self) -> connection:
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            connect_timeout=3,
            options="-c statement_timeout=15000 -c default_transaction_read_only=on",
        )

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        record_id: str | None = None,
        record_ids: list[str] | None = None,
        split: str | None = None,
        company_identifier: str | None = None,
    ) -> VectorSearchOutput:
        from agent.tools._clients import get_sentence_encoder

        if record_ids == []:
            return VectorSearchOutput(query=query_text, chunks=[], total_found=0)
        company = (company_identifier or "").strip().upper()
        with self._encoder_lock:
            encoder = get_sentence_encoder(self.model_name)
            query_emb = embedding_values(
                encoder.encode(query_text, normalize_embeddings=True)
            )

        with closing(self._get_connection()) as conn, conn:
            rows = search_embeddings(
                conn,
                query_emb,
                self.model_name,
                top_k,
                record_id=record_id,
                record_ids=record_ids,
                company=None if company in {"UNKNOWN", "N/A", ""} else company,
                split=split,
            )

        records = []
        for rid, filename, section, index, contents, score, start, end, version in rows:
            gold_ids = (
                [f"text_{position}" for position in range(start, end + 1)]
                if version == "2" and start is not None and end is not None
                else []
            )
            records.append(
                VectorChunkRecord(
                    record_id=rid,
                    filename=filename,
                    section=section,
                    chunk_index=index,
                    text_content=contents,
                    similarity_score=float(score),
                    evidence_id=f"chunk::{rid}::{section}::{index}",
                    page=source_page(filename or ""),
                    sentence_start=start,
                    sentence_end=end,
                    gold_ids=gold_ids,
                )
            )

        return VectorSearchOutput(
            query=query_text,
            chunks=records,
            total_found=len(records),
        )


def vector_retrieval_tool(
    payload: VectorSearchInput, client: VectorRetrievalTool | None = None
) -> ToolResult[VectorSearchOutput]:
    """Instrumented tool entrypoint for semantic vector retrieval."""
    if client is None:

        from agent.tools._clients import get_vector_tool_client

        instance: VectorRetrievalTool = get_vector_tool_client()
    else:
        instance = client
    return ToolResult.execute_instrumented(
        tool_name="vector_retrieval",
        fn=instance.search,
        query_text=payload.query_text,
        top_k=payload.top_k,
        record_id=payload.record_id,
        record_ids=payload.record_ids,
        company_identifier=payload.company_identifier,
        split=payload.split,
    )
