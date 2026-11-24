"""Upload pipeline: PDF bytes -> pages -> chunks -> embeddings -> Postgres."""

import hashlib
from dataclasses import dataclass
from uuid import UUID

import psycopg

from app.chunking import chunk_pages, extract_pages
from app.config import get_settings
from app.embeddings import EMBED_BATCH, Embedder


class DuplicateDocument(Exception):
    def __init__(self, existing_id: UUID):
        self.existing_id = existing_id


class EmptyDocument(Exception):
    pass


@dataclass
class IngestResult:
    id: UUID
    filename: str
    pages: int
    chunks: int


def ingest_pdf(
    conn: psycopg.Connection, embedder: Embedder, filename: str, data: bytes
) -> IngestResult:
    settings = get_settings()
    digest = hashlib.sha256(data).hexdigest()
    existing = conn.execute("SELECT id FROM documents WHERE sha256 = %s", (digest,)).fetchone()
    if existing:
        raise DuplicateDocument(existing[0])

    pages = extract_pages(data)  # ValueError for a broken file
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise EmptyDocument()

    vectors: list[list[float]] = []
    for i in range(0, len(chunks), EMBED_BATCH):
        vectors.extend(embedder.embed_documents([c.text for c in chunks[i : i + EMBED_BATCH]]))

    doc_id = conn.execute(
        "INSERT INTO documents (filename, pages, sha256) VALUES (%s, %s, %s) RETURNING id",
        (filename, len(pages), digest),
    ).fetchone()[0]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (document_id, page, chunk_index, content, embedding)"
            " VALUES (%s, %s, %s, %s, %s::vector)",
            [
                (doc_id, c.page, i, c.text, str(vec))
                for i, (c, vec) in enumerate(zip(chunks, vectors, strict=True))
            ],
        )
    return IngestResult(id=doc_id, filename=filename, pages=len(pages), chunks=len(chunks))
