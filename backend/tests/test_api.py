"""API tests against a real Postgres + pgvector (docker compose up -d).

They use the offline hash embedder and the fake LLM, so no key or model download is needed."""

import uuid

import pytest
from pdfgen import make_pdf

pytestmark = pytest.mark.integration

GUIDE = make_pdf(
    [
        [
            "DocuMind onboarding guide.",
            "The office opens at 9 am and closes at 6 pm on weekdays.",
        ],
        [
            "Leave policy.",
            "Every employee gets 24 days of paid leave per year.",
            "Unused leave can be carried over for one year only.",
        ],
    ]
)
RECIPES = make_pdf([["Banana bread needs three ripe bananas and two cups of flour."]])


@pytest.fixture(scope="module")
def uploaded(client):
    ids = {}
    for name, data in (("guide.pdf", GUIDE), ("recipes.pdf", RECIPES)):
        response = client.post("/api/documents", files={"file": (name, data, "application/pdf")})
        assert response.status_code == 201, response.text
        ids[name] = response.json()["id"]
    return ids


def test_health(client):
    body = client.get("/api/health").json()
    assert body == {"status": "ok", "database": "ok", "llm_provider": "fake"}


def test_upload_returns_counts(client, uploaded):
    docs = {d["filename"]: d for d in client.get("/api/documents").json()}
    assert docs["guide.pdf"]["pages"] == 2
    assert docs["guide.pdf"]["chunks"] >= 2


def test_duplicate_upload_is_409(client, uploaded):
    response = client.post(
        "/api/documents", files={"file": ("again.pdf", GUIDE, "application/pdf")}
    )
    assert response.status_code == 409


def test_non_pdf_is_415(client):
    response = client.post("/api/documents", files={"file": ("a.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_pdf_without_text_is_422(client):
    response = client.post(
        "/api/documents", files={"file": ("blank.pdf", make_pdf([[]]), "application/pdf")}
    )
    assert response.status_code == 422


def test_ask_finds_the_right_page_and_cites_it(client, uploaded):
    response = client.post("/api/ask", json={"question": "How many days of paid leave per year?"})
    assert response.status_code == 200, response.text
    body = response.json()
    top = body["sources"][0]
    assert top["filename"] == "guide.pdf" and top["page"] == 2
    assert body["citations"] == [1]
    assert top["cited"] is True
    assert "[1]" in body["answer"]


def test_ask_can_be_limited_to_one_document(client, uploaded):
    response = client.post(
        "/api/ask",
        json={"question": "bananas flour", "document_ids": [uploaded["recipes.pdf"]]},
    )
    assert {s["filename"] for s in response.json()["sources"]} == {"recipes.pdf"}


@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid", "hybrid_rerank"])
def test_every_retrieval_mode_works(client, uploaded, mode):
    response = client.post("/api/ask", json={"question": "paid leave per year", "mode": mode})
    assert response.status_code == 200
    assert response.json()["sources"][0]["filename"] == "guide.pdf"


def test_keyword_search_matches_natural_questions(client, uploaded):
    """Regression: an AND query needed every question word in one chunk and found nothing."""
    response = client.post(
        "/api/ask",
        json={"question": "How many days of paid leave does an employee get?", "mode": "keyword"},
    )
    sources = response.json()["sources"]
    assert sources and sources[0]["filename"] == "guide.pdf" and sources[0]["page"] == 2


def test_question_of_only_stopwords_does_not_crash(client, uploaded):
    response = client.post("/api/ask", json={"question": "is it the", "mode": "keyword"})
    assert response.status_code == 200


@pytest.mark.parametrize(
    "body",
    [{"question": ""}, {"question": "x", "top_k": 0}, {"question": "x", "mode": "magic"}, {}],
)
def test_bad_ask_bodies_are_422(client, body):
    assert client.post("/api/ask", json=body).status_code == 422


def test_delete_document(client):
    doc = make_pdf([["Temporary document about zebras."]])
    doc_id = client.post(
        "/api/documents", files={"file": ("tmp.pdf", doc, "application/pdf")}
    ).json()["id"]
    assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404
    assert client.delete(f"/api/documents/{uuid.uuid4()}").status_code == 404
    assert client.delete("/api/documents/not-a-uuid").status_code == 422
