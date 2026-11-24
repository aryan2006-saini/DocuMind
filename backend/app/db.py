"""Postgres + pgvector access. Plain SQL through psycopg 3."""

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector

from app.config import get_settings


def schema_sql(dim: int) -> str:
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    filename    text NOT NULL,
    pages       int  NOT NULL,
    sha256      text NOT NULL UNIQUE,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    document_id  uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page         int  NOT NULL,
    chunk_index  int  NOT NULL,
    content      text NOT NULL,
    embedding    vector({dim}) NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_document_id ON chunks (document_id);
"""


def init_schema() -> None:
    settings = get_settings()
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        conn.execute(schema_sql(settings.embed_dim))


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    """One connection per request; commits on success, rolls back on error."""
    with psycopg.connect(get_settings().database_url) as conn:
        register_vector(conn)
        yield conn


def get_conn() -> Iterator[psycopg.Connection]:
    """FastAPI dependency."""
    with connect() as conn:
        yield conn
