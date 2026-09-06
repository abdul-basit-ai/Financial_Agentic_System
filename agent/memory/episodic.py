"""PostgreSQL pgvector episodic memory store with exponential recency decay."""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any
import psycopg2
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_HALF_LIFE_DAYS = 30.0

EPISODIC_SCHEMA_DDL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS episodic_memories (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    company_identifier TEXT,
    query_text TEXT NOT NULL,
    summary TEXT NOT NULL,
    key_findings JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    importance_score FLOAT NOT NULL DEFAULT 0.5,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    embedding vector({EMBEDDING_DIM})
);

CREATE INDEX IF NOT EXISTS idx_episodic_session ON episodic_memories (session_id);
CREATE INDEX IF NOT EXISTS idx_episodic_company ON episodic_memories (company_identifier);

-- NOTE: ivfflat index is NOT created here. With lists=50 it silently returns
-- zero rows on small tables (empty index clusters until populated). The
-- ensure_vector_index() helper builds it lazily once the table has enough rows.
"""


class EpisodicEntry(BaseModel):
    session_id: str
    trace_id: str
    query_text: str
    summary: str
    company_identifier: str | None = None
    key_findings: dict[str, Any] = Field(default_factory=dict)
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EpisodicRetrievalResult(BaseModel):
    id: int
    session_id: str
    query_text: str
    summary: str
    company_identifier: str | None
    key_findings: dict[str, Any]
    importance_score: float
    semantic_similarity: float
    recency_score: float
    final_score: float


def compute_recency_score(created_at: datetime, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> float:
    """Calculates exponential memory decay: S_rec = 2^(-delta_t / tau_half)."""
    now = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    delta_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
    return math.pow(2.0, -delta_days / half_life_days)


class EpisodicMemoryStore:
    """Stores and retrieves historical analytical episodes using multi-factor decay."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        dbname: str | None = None,
        user: str | None = None,
        password: str | None = None,
        model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        self.conn_params = {
            "host": host or os.getenv("POSTGRES_HOST", "localhost"),
            "port": port or int(os.getenv("POSTGRES_PORT", "5432")),
            "dbname": dbname or os.getenv("POSTGRES_DB", "financial_agent"),
            "user": user or os.getenv("POSTGRES_USER", "postgres"),
            "password": password or os.getenv("POSTGRES_PASSWORD", "password"),
        }
        self.model = SentenceTransformer(model_name)
        self._conn = None

    def _get_connection(self) -> Any:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(**self.conn_params)
        return self._conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
            self._conn = None

    def init_schema(self) -> None:
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(EPISODIC_SCHEMA_DDL)
        conn.commit()

    def ensure_vector_index(self, min_rows: int = 1000, lists: int = 50) -> bool:
        """Creates the ivfflat index once the table is large enough to populate it.

        Returns True if the index exists (was created now or previously).
        Below min_rows the index would silently return zero results, so exact
        sequential scan is used instead (correct, just slower at small scale).
        """
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM episodic_memories")
            count = int(cur.fetchone()[0])
            if count < min_rows:
                cur.execute("DROP INDEX IF EXISTS idx_episodic_vector")
                conn.commit()
                return False
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_episodic_vector
                ON episodic_memories USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {int(lists)});
                """
            )
            conn.commit()
            return True

    def save_episode(self, entry: EpisodicEntry) -> int:
        conn = self._get_connection()
        embedding = self.model.encode(
            f"{entry.query_text} {entry.summary}", normalize_embeddings=True
        ).tolist()

        sql = """
        INSERT INTO episodic_memories (
            session_id, trace_id, company_identifier, query_text, summary,
            key_findings, importance_score, created_at, embedding
        ) VALUES (
            %(session_id)s, %(trace_id)s, %(company)s, %(query)s, %(summary)s,
            %(findings)s, %(importance)s, %(created_at)s, %(embedding)s::vector
        ) RETURNING id;
        """
        params = {
            "session_id": entry.session_id,
            "trace_id": entry.trace_id,
            "company": entry.company_identifier,
            "query": entry.query_text,
            "summary": entry.summary,
            "findings": psycopg2.extras.Json(entry.key_findings),
            "importance": entry.importance_score,
            "created_at": entry.created_at,
            "embedding": str(embedding),
        }

        with conn.cursor() as cur:
            cur.execute(sql, params)
            new_id = cur.fetchone()[0]
        conn.commit()
        return int(new_id)

    def retrieve_episodes(
        self,
        query: str,
        company_identifier: str | None = None,
        top_k: int = 5,
        w_sim: float = 0.5,
        w_rec: float = 0.3,
        w_imp: float = 0.2,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    ) -> list[EpisodicRetrievalResult]:
        conn = self._get_connection()
        query_emb = self.model.encode(query, normalize_embeddings=True).tolist()

        sql = """
        SELECT
            id, session_id, trace_id, company_identifier, query_text, summary,
            key_findings, importance_score, created_at,
            1.0 - (embedding <=> %(embedding)s::vector) AS sim
        FROM episodic_memories
        WHERE (%(company)s IS NULL OR company_identifier = %(company)s)
        ORDER BY embedding <=> %(embedding)s::vector
        LIMIT %(candidate_limit)s;
        """
        params = {
            "embedding": str(query_emb),
            "company": company_identifier,
            "candidate_limit": top_k * 3,
        }

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            candidates = cur.fetchall()

        results: list[EpisodicRetrievalResult] = []
        for c in candidates:
            s_sim = max(0.0, float(c["sim"]))
            s_rec = compute_recency_score(c["created_at"], half_life_days)
            s_imp = float(c["importance_score"])
            final_score = (w_sim * s_sim) + (w_rec * s_rec) + (w_imp * s_imp)

            results.append(
                EpisodicRetrievalResult(
                    id=c["id"],
                    session_id=c["session_id"],
                    query_text=c["query_text"],
                    summary=c["summary"],
                    company_identifier=c["company_identifier"],
                    key_findings=c["key_findings"],
                    importance_score=s_imp,
                    semantic_similarity=round(s_sim, 4),
                    recency_score=round(s_rec, 4),
                    final_score=round(final_score, 4),
                )
            )

        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:top_k]