# DocuMind — RAG Document Assistant

Upload PDFs, ask questions in plain English, and get answers that cite the exact page they came from.

[![CI](https://github.com/aryan2006-saini/DocuMind/actions/workflows/ci.yml/badge.svg)](https://github.com/aryan2006-saini/DocuMind/actions/workflows/ci.yml) ![stack](https://img.shields.io/badge/FastAPI-Python_3.13-009688) ![db](https://img.shields.io/badge/Postgres-pgvector-336791) ![ui](https://img.shields.io/badge/React-TypeScript-61dafb) ![llm](https://img.shields.io/badge/LLM-Groq_Llama_3.3-f55036)

## How it works

```
PDF ──► pypdf ──► page-aware chunks (900 chars, 150 overlap, sentence boundaries)
                      │
                      ▼
          local embeddings (BAAI/bge-small-en-v1.5, ONNX via fastembed)
                      │
                      ▼
      Postgres: pgvector HNSW index  +  full-text tsvector GIN index

Question ──► vector search (top 20) ─┐
         └─► full-text search (top 20) ┴─► Reciprocal Rank Fusion
                                            │
                                            ▼
                          cross-encoder rerank (ms-marco-MiniLM) ──► top 5
                                            │
                                            ▼
                Groq Llama 3.3: answer ONLY from numbered sources, cite [n]
                                            │
                                            ▼
                answer + citations (file, page, snippet) in the React UI
```

- **Hybrid retrieval.** Vector search finds meaning ("vacation" ↔ "paid leave"); keyword search finds exact terms (names, codes, numbers). RRF merges both rankings.
- **Reranking.** A cross-encoder reads the question and each candidate together for a sharper final order.
- **Grounded answers.** The prompt allows only the given sources, demands `[n]` citations, and has a fixed "not in the documents" reply. Citations that point to sources that don't exist are dropped.
- **Evaluation.** `eval/run_eval.py` measures hit@k and MRR for each retrieval mode, plus LLM-judged faithfulness and citation rate.
- **Runs locally.** Embeddings and reranking run on your CPU. Only answer generation calls Groq (free tier).

## Run it

Needs Python 3.12+ with [uv](https://docs.astral.sh/uv/), Node 20+, and Docker Desktop.

```bash
cp .env.example .env          # then paste your free key from https://console.groq.com/keys
docker compose up -d          # Postgres + pgvector on localhost:5433

cd backend
uv sync
uv run uvicorn app.main:app --reload     # API on http://127.0.0.1:8000  (docs at /docs)

cd ../frontend                # in a second terminal
npm install
npm run dev                   # UI on http://localhost:5173
```

The first upload downloads the embedding and reranking models (about 150 MB) into `.fastembed_cache/`.

No Groq key yet? Set `LLM_PROVIDER=fake` in `.env` to try everything offline. The fake model just quotes the top source.

## API

| Method | Path | What it does |
|---|---|---|
| `GET` | `/api/health` | API, database and LLM provider status |
| `POST` | `/api/documents` | Upload a PDF (multipart `file`). `409` duplicate, `413` too big, `415` not a PDF, `422` no text |
| `GET` | `/api/documents` | List documents with page and chunk counts |
| `DELETE` | `/api/documents/{id}` | Delete a document and its chunks |
| `POST` | `/api/ask` | `{question, top_k?, document_ids?, mode?}` → answer, citations, sources, latency |

`mode` is one of `vector`, `keyword`, `hybrid`, `hybrid_rerank` (default).

## Tests

```bash
cd backend
uv run pytest -m "not integration"   # 21 fast tests, no database
uv run pytest                        # all 39, needs `docker compose up -d`
uv run ruff check .
uv run ruff format --check .

cd ../frontend
npm run format:check            # Prettier
```

The API tests use their own `documind_test` database, an offline hash embedder and the fake LLM, so they need no key and no model download.

## Evaluate

1. Upload the PDFs you want to test with.
2. Copy `eval/questions.example.jsonl` to `eval/questions.jsonl` and write 30–50 questions. Give each one the file name and the page(s) that hold the answer.
3. Run:

```bash
cd backend
uv run python -m eval.run_eval --questions ../eval/questions.jsonl           # retrieval only
uv run python -m eval.run_eval --questions ../eval/questions.jsonl --judge   # + faithfulness
```

It prints a table like the one below and saves JSON to `eval/results/`:

| mode | hit@5 | MRR |
|---|---|---|
| vector | … | … |
| keyword | … | … |
| hybrid | … | … |
| hybrid_rerank | … | … |

## Project layout

```
backend/app/
  main.py        FastAPI routes and request/response models
  chunking.py    PDF text extraction and sentence-aware chunking
  embeddings.py  fastembed embedder + cross-encoder reranker (and an offline hash embedder)
  retrieval.py   vector search, full-text search, RRF fusion, reranking
  rag.py         grounded prompt and citation parsing
  llm.py         Groq client (OpenAI-compatible API) and the offline fake LLM
  ingest.py      upload pipeline: dedupe by SHA-256, chunk, embed, store
  db.py          schema (pgvector HNSW + tsvector GIN) and connections
backend/eval/    evaluation script
backend/tests/   unit and API tests
frontend/src/    React + TypeScript UI
```
