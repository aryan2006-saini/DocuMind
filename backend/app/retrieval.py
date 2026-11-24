"""Hybrid retrieval: pgvector similarity + Postgres full-text search, fused with
Reciprocal Rank Fusion (RRF), then optionally reranked by a cross-encoder."""

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

import psycopg

from app.embeddings import Embedder, Reranker


class Mode(StrEnum):
    vector = "vector"
    keyword = "keyword"
    hybrid = "hybrid"
    hybrid_rerank = "hybrid_rerank"


@dataclass
class Hit:
    chunk_id: int
    document_id: UUID
    filename: str
    page: int
    content: str
    score: float


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank). Best first.

    Ties keep the order in which ids were first seen, so the result is deterministic.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _doc_filter(document_ids: list[UUID] | None) -> tuple[str, tuple]:
    if not document_ids:
        return "", ()
    return " AND c.document_id = ANY(%s)", (list(document_ids),)


def vector_search(
    conn: psycopg.Connection, query_vec: list[float], limit: int, document_ids=None
) -> list[int]:
    where, params = _doc_filter(document_ids)
    rows = conn.execute(
        f"SELECT c.id FROM chunks c WHERE true{where}"
        " ORDER BY c.embedding <=> %s::vector LIMIT %s",
        (*params, str(query_vec), limit),
    ).fetchall()
    return [r[0] for r in rows]


def keyword_search(
    conn: psycopg.Connection, question: str, limit: int, document_ids=None
) -> list[int]:
    where, params = _doc_filter(document_ids)
    # Match chunks containing ANY of the question's words (OR), then rank by coverage and
    # density. An AND query (websearch_to_tsquery) almost never matches a full question.
    rows = conn.execute(
        "SELECT c.id FROM chunks c,"
        " to_tsquery('english', replace(plainto_tsquery('english', %s)::text, '&', '|')) q"
        f" WHERE c.tsv @@ q{where}"
        " ORDER BY ts_rank_cd(c.tsv, q) DESC LIMIT %s",
        (question, *params, limit),
    ).fetchall()
    return [r[0] for r in rows]


def _load(conn: psycopg.Connection, ids: list[int]) -> dict[int, Hit]:
    if not ids:
        return {}
    rows = conn.execute(
        "SELECT c.id, c.document_id, d.filename, c.page, c.content"
        " FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.id = ANY(%s)",
        (ids,),
    ).fetchall()
    return {r[0]: Hit(r[0], r[1], r[2], r[3], r[4], 0.0) for r in rows}


def retrieve(
    conn: psycopg.Connection,
    embedder: Embedder,
    reranker: Reranker | None,
    question: str,
    top_k: int,
    candidate_k: int,
    mode: Mode = Mode.hybrid_rerank,
    document_ids: list[UUID] | None = None,
) -> list[Hit]:
    rankings: list[list[int]] = []
    if mode != Mode.keyword:
        rankings.append(
            vector_search(conn, embedder.embed_query(question), candidate_k, document_ids)
        )
    if mode != Mode.vector:
        rankings.append(keyword_search(conn, question, candidate_k, document_ids))

    fused = rrf(rankings)
    hits_by_id = _load(conn, [chunk_id for chunk_id, _ in fused])
    hits = []
    for chunk_id, score in fused:
        if chunk_id in hits_by_id:
            hit = hits_by_id[chunk_id]
            hit.score = score
            hits.append(hit)

    if mode == Mode.hybrid_rerank and reranker is not None and len(hits) > 1:
        for hit, score in zip(hits, reranker.scores(question, [h.content for h in hits])):
            hit.score = score
        hits.sort(key=lambda h: h.score, reverse=True)

    return hits[:top_k]
