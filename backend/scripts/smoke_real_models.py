"""Check the real local models end to end (no LLM key needed): embed, rerank, retrieve.

uv run python scripts/smoke_real_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.embeddings import CrossEncoderReranker, FastEmbedder

settings = get_settings()
embedder = FastEmbedder(settings.embed_model, settings.embed_dim)
docs = [
    "Employees receive 24 days of paid leave every year.",
    "Banana bread needs three ripe bananas.",
    "The office opens at 9 am on weekdays.",
]
vectors = embedder.embed_documents(docs)
query = embedder.embed_query("How much vacation do staff get?")
assert len(query) == settings.embed_dim, len(query)
similarities = [sum(a * b for a, b in zip(query, v, strict=True)) for v in vectors]
print("embedding similarities:", [round(s, 3) for s in similarities])
assert max(range(3), key=lambda i: similarities[i]) == 0

reranker = CrossEncoderReranker(settings.rerank_model)
scores = reranker.scores("How much vacation do staff get?", docs)
print("rerank scores:", [round(s, 3) for s in scores])
assert max(range(3), key=lambda i: scores[i]) == 0
print("OK: real embedding and reranking models work")
