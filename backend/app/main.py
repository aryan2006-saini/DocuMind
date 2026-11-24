"""DocuMind API."""

import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated
from uuid import UUID

import psycopg
from fastapi import Depends, FastAPI, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import db
from app.config import get_settings
from app.embeddings import Embedder, Reranker, get_embedder, get_reranker
from app.ingest import DuplicateDocument, EmptyDocument, ingest_pdf
from app.llm import LLM, LLMError, get_llm
from app.rag import NO_ANSWER, SYSTEM_PROMPT, build_prompt, parse_citations
from app.retrieval import Mode, retrieve


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_schema()
    yield


app = FastAPI(title="DocuMind", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in get_settings().cors_origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

Conn = Annotated[psycopg.Connection, Depends(db.get_conn)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
RerankerDep = Annotated[Reranker | None, Depends(get_reranker)]


def llm_dependency() -> LLM:
    try:
        return get_llm()
    except LLMError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


LLMDep = Annotated[LLM, Depends(llm_dependency)]


# ---------- schemas ----------
class DocumentOut(BaseModel):
    id: UUID
    filename: str
    pages: int
    chunks: int
    created_at: datetime | None = None


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=10)
    document_ids: list[UUID] | None = None
    mode: Mode = Mode.hybrid_rerank


class SourceOut(BaseModel):
    n: int
    document_id: UUID
    filename: str
    page: int
    snippet: str
    score: float
    cited: bool


class AskOut(BaseModel):
    answer: str
    citations: list[int]
    sources: list[SourceOut]
    model: str
    latency_ms: int


# ---------- routes ----------
@app.get("/api/health")
def health():
    try:
        with db.connect() as conn:
            conn.execute("SELECT 1")
        database = "ok"
    except psycopg.Error:
        database = "unavailable"
    return {"status": "ok", "database": database, "llm_provider": get_settings().llm_provider}


@app.post("/api/documents", status_code=status.HTTP_201_CREATED, response_model=DocumentOut)
async def upload_document(file: UploadFile, conn: Conn, embedder: EmbedderDep):
    settings = get_settings()
    filename = (file.filename or "document.pdf").strip()[:255]
    data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB.")
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "Only PDF files are supported.")
    try:
        result = ingest_pdf(conn, embedder, filename, data)
    except DuplicateDocument as exc:
        raise HTTPException(409, f"This PDF is already uploaded (id {exc.existing_id}).") from exc
    except EmptyDocument as exc:
        raise HTTPException(422, "No text found in this PDF. Scanned PDFs need OCR first.") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return DocumentOut(**result.__dict__)


@app.get("/api/documents", response_model=list[DocumentOut])
def list_documents(conn: Conn):
    rows = conn.execute(
        "SELECT d.id, d.filename, d.pages, count(c.id), d.created_at"
        " FROM documents d LEFT JOIN chunks c ON c.document_id = d.id"
        " GROUP BY d.id ORDER BY d.created_at DESC"
    ).fetchall()
    return [
        DocumentOut(id=r[0], filename=r[1], pages=r[2], chunks=r[3], created_at=r[4]) for r in rows
    ]


@app.delete("/api/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: UUID, conn: Conn):
    deleted = conn.execute("DELETE FROM documents WHERE id = %s", (document_id,)).rowcount
    if not deleted:
        raise HTTPException(404, "Document not found.")


@app.post("/api/ask", response_model=AskOut)
def ask(body: AskIn, conn: Conn, embedder: EmbedderDep, reranker: RerankerDep, llm: LLMDep):
    settings = get_settings()
    started = time.perf_counter()
    hits = retrieve(
        conn,
        embedder,
        reranker,
        body.question,
        top_k=body.top_k or settings.top_k,
        candidate_k=settings.candidate_k,
        mode=body.mode,
        document_ids=body.document_ids,
    )
    if not hits:
        answer, cited = NO_ANSWER, []
    else:
        try:
            answer = llm.complete(SYSTEM_PROMPT, build_prompt(body.question, hits))
        except LLMError as exc:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
        cited = parse_citations(answer, len(hits))

    sources = [
        SourceOut(
            n=n,
            document_id=h.document_id,
            filename=h.filename,
            page=h.page,
            snippet=h.content[:500],
            score=round(h.score, 4),
            cited=n in cited,
        )
        for n, h in enumerate(hits, 1)
    ]
    return AskOut(
        answer=answer,
        citations=cited,
        sources=sources,
        model=llm.name,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
