import os

import psycopg
import pytest

# Tests never call Groq or download models, and use their own database.
ADMIN_URL = os.environ.get(
    "DOCUMIND_ADMIN_URL", "postgresql://documind:documind@localhost:5433/documind"
)
TEST_URL = ADMIN_URL.rsplit("/", 1)[0] + "/documind_test"
os.environ["DATABASE_URL"] = TEST_URL
os.environ["LLM_PROVIDER"] = "fake"
os.environ["EMBED_PROVIDER"] = "hash"
os.environ["USE_RERANKER"] = "false"


def _database_available() -> bool:
    try:
        with psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=3) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = 'documind_test'"
            ).fetchone()
            if not exists:
                conn.execute("CREATE DATABASE documind_test")
        return True
    except psycopg.Error:
        return False


@pytest.fixture(scope="session")
def client():
    if not _database_available():
        pytest.skip("Postgres is not running: start it with `docker compose up -d`")
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:  # runs the lifespan, which creates the schema
        with psycopg.connect(TEST_URL, autocommit=True) as conn:
            conn.execute("TRUNCATE documents CASCADE")
        yield test_client
