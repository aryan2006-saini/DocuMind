"""Local embeddings and reranking. Nothing here calls a paid API."""

import hashlib
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from app.config import get_settings

# Models download into the project folder (git-ignored), not a system-wide cache.
MODEL_CACHE = str(Path(__file__).resolve().parents[2] / ".fastembed_cache")


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    """ONNX sentence embeddings via fastembed (no PyTorch). The model downloads on first use."""

    def __init__(self, model_name: str, dim: int):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name, cache_dir=MODEL_CACHE)
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # query_embed adds the query instruction that BGE models expect
        return next(iter(self.model.query_embed(text))).tolist()


class HashEmbedder:
    """Tiny deterministic bag-of-words embedder for tests and offline demos."""

    def __init__(self, dim: int):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int.from_bytes(hashlib.md5(word.encode()).digest()[:4], "little") % self.dim
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class Reranker(Protocol):
    def scores(self, query: str, documents: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Cross-encoder that reads the question and each chunk together and scores relevance."""

    def __init__(self, model_name: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(model_name=model_name, cache_dir=MODEL_CACHE)

    def scores(self, query: str, documents: list[str]) -> list[float]:
        return [float(s) for s in self.model.rerank(query, documents)]


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    if settings.embed_provider == "hash":
        return HashEmbedder(settings.embed_dim)
    return FastEmbedder(settings.embed_model, settings.embed_dim)


@lru_cache
def get_reranker() -> Reranker | None:
    settings = get_settings()
    if not settings.use_reranker or settings.embed_provider == "hash":
        return None
    return CrossEncoderReranker(settings.rerank_model)
