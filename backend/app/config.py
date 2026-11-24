from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings come from environment variables or the repo-root .env file."""

    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")

    database_url: str = "postgresql://documind:documind@localhost:5433/documind"

    # LLM
    llm_provider: str = "groq"  # groq | fake
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    llm_timeout_s: float = 60.0

    # Embeddings and reranking
    embed_provider: str = "fastembed"  # fastembed | hash
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    use_reranker: bool = True
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Chunking and retrieval
    chunk_size: int = 900  # characters
    chunk_overlap: int = 150
    top_k: int = 5  # chunks sent to the LLM
    candidate_k: int = 20  # chunks each retriever returns before fusion

    max_upload_mb: int = 25
    cors_origins: str = "http://localhost:5173"


@lru_cache
def get_settings() -> Settings:
    return Settings()
