"""Fast tests: no database, no network."""

import pytest
from pdfgen import make_pdf

from app.chunking import chunk_pages, clean_text, extract_pages, split_text
from app.embeddings import HashEmbedder
from app.llm import FakeLLM
from app.rag import build_prompt, parse_citations
from app.retrieval import Hit, rrf


# ---------- chunking ----------
def test_short_text_is_one_chunk():
    assert split_text("Hello world.", 100, 10) == ["Hello world."]


def test_empty_text_has_no_chunks():
    assert split_text("   \n\n ", 100, 10) == []


def test_chunks_respect_size_and_cover_all_text():
    sentences = [f"Sentence number {i} talks about topic {i}." for i in range(60)]
    text = " ".join(sentences)
    chunks = split_text(text, 200, 40)
    assert all(len(c) <= 200 for c in chunks)
    joined = " ".join(chunks)
    for s in sentences:
        assert s in joined


def test_chunks_prefer_sentence_boundaries():
    text = " ".join(f"This is sentence {i}." for i in range(50))
    chunks = split_text(text, 120, 20)
    assert all(c.endswith(".") for c in chunks[:-1])


def test_chunks_overlap():
    text = " ".join(f"word{i}" for i in range(200))
    first, second = split_text(text, 100, 30)[:2]
    assert first.split()[-1] in second


def test_bad_overlap_is_rejected():
    with pytest.raises(ValueError):
        split_text("abc", 10, 10)


def test_clean_text_rejoins_hyphenated_words():
    assert clean_text("retrie-\nval   augmented") == "retrieval augmented"


def test_chunk_pages_keeps_page_numbers():
    chunks = chunk_pages(["Page one text.", "", "Page three text."], 100, 10)
    assert [(c.page, c.text) for c in chunks] == [(1, "Page one text."), (3, "Page three text.")]


def test_extract_pages_reads_generated_pdf():
    pages = extract_pages(make_pdf([["Alpha line"], ["Beta line"]]))
    assert len(pages) == 2
    assert "Alpha" in pages[0] and "Beta" in pages[1]


def test_extract_pages_rejects_garbage():
    with pytest.raises(ValueError):
        extract_pages(b"%PDF-1.4 this is not really a pdf")


# ---------- fusion ----------
def test_rrf_rewards_items_found_by_both_retrievers():
    fused = rrf([[1, 2, 3], [3, 4, 1]])
    order = [chunk_id for chunk_id, _ in fused]
    assert order[0] in (1, 3)
    assert set(order[:2]) == {1, 3}
    assert set(order) == {1, 2, 3, 4}


def test_rrf_single_list_keeps_order():
    assert [c for c, _ in rrf([[5, 6, 7]])] == [5, 6, 7]


def test_rrf_empty():
    assert rrf([]) == []


# ---------- citations and prompts ----------
@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Fact one [1]. Fact two [2][3].", [1, 2, 3]),
        ("Both [2, 1].", [2, 1]),
        ("Repeated [1] and again [1].", [1]),
        ("Made-up source [9] and [0].", []),
        ("No citations at all.", []),
    ],
)
def test_parse_citations(answer, expected):
    assert parse_citations(answer, 3) == expected


def _hit(n: int, text: str) -> Hit:
    return Hit(chunk_id=n, document_id=None, filename="guide.pdf", page=n, content=text, score=0)


def test_build_prompt_numbers_sources_with_page():
    prompt = build_prompt("What?", [_hit(1, "A."), _hit(2, "B.")])
    assert "[1] guide.pdf, page 1\nA." in prompt
    assert "[2] guide.pdf, page 2\nB." in prompt
    assert prompt.endswith("Question: What?")


def test_fake_llm_cites_first_source():
    answer = FakeLLM().complete("", build_prompt("Q", [_hit(1, "First fact. More."), _hit(2, "X")]))
    assert answer == "First fact. [1]"


# ---------- embeddings ----------
def dot(x: list[float], y: list[float]) -> float:
    return sum(i * j for i, j in zip(x, y, strict=True))


def test_hash_embedder_is_normalised_and_similar_for_similar_text():
    embedder = HashEmbedder(64)
    a, b, c = embedder.embed_documents(["vector database index", "vector index", "banana bread"])
    assert abs(dot(a, a) - 1) < 1e-9
    assert dot(a, b) > dot(a, c)
