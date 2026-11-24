"""Measure retrieval and answer quality on your own question set.

Usage (from backend/, after uploading the PDFs the questions are about):

    uv run python -m eval.run_eval --questions ../eval/questions.jsonl
    uv run python -m eval.run_eval --questions ../eval/questions.jsonl --judge   # + faithfulness

Each line of the questions file is JSON:
    {"question": "How many days of paid leave?", "doc": "handbook.pdf", "pages": [12]}

Metrics
- hit@k: share of questions where a chunk from the right document AND one of the right pages
  is in the top k results.
- MRR: mean of 1 / rank of the first correct chunk (0 if none in the top k).
- citation rate (--judge): share of answers that cite at least one source.
- faithfulness (--judge): share of answers the LLM judge marks fully supported by the sources.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from app import db
from app.config import get_settings
from app.embeddings import get_embedder, get_reranker
from app.llm import LLMError, get_llm
from app.rag import NO_ANSWER, SYSTEM_PROMPT, build_prompt, parse_citations
from app.retrieval import Hit, Mode, retrieve

JUDGE_PROMPT = """You check whether an answer is fully supported by the given sources.
Reply with exactly one word: SUPPORTED if every factual claim in the answer is stated in the
sources (or the answer says the information is not in the documents), otherwise UNSUPPORTED."""


def load_questions(path: Path) -> list[dict]:
    items = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        item = json.loads(line)
        if not {"question", "doc", "pages"} <= item.keys():
            raise SystemExit(f"line {number}: needs question, doc and pages")
        items.append(item)
    return items


def first_correct_rank(hits: list[Hit], doc: str, pages: list[int]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if hit.filename == doc and hit.page in pages:
            return rank
    return None


def evaluate_retrieval(conn, questions, modes, k):
    settings = get_settings()
    embedder, reranker = get_embedder(), get_reranker()
    results = {}
    for mode in modes:
        ranks = []
        for q in questions:
            hits = retrieve(
                conn, embedder, reranker, q["question"], k, settings.candidate_k, mode=mode
            )
            ranks.append(first_correct_rank(hits, q["doc"], q["pages"]))
        hit_rate = sum(r is not None for r in ranks) / len(ranks)
        mrr = sum(1 / r for r in ranks if r) / len(ranks)
        results[mode.value] = {"hit_rate": round(hit_rate, 3), "mrr": round(mrr, 3)}
    return results


def evaluate_answers(conn, questions, k, delay_s):
    settings = get_settings()
    embedder, reranker, llm = get_embedder(), get_reranker(), get_llm()
    supported = cited = 0
    details = []
    for q in questions:
        hits = retrieve(conn, embedder, reranker, q["question"], k, settings.candidate_k)
        prompt = build_prompt(q["question"], hits)
        answer = llm.complete(SYSTEM_PROMPT, prompt) if hits else NO_ANSWER
        verdict = llm.complete(JUDGE_PROMPT, f"{prompt}\n\nAnswer to check:\n{answer}")
        is_supported = verdict.strip().upper().startswith("SUPPORTED")
        has_citation = bool(parse_citations(answer, len(hits)))
        supported += is_supported
        cited += has_citation
        details.append(
            {"question": q["question"], "answer": answer, "supported": is_supported,
             "cited": has_citation}
        )
        time.sleep(delay_s)  # stay inside the Groq free-tier rate limit
    n = len(questions)
    return {
        "model": llm.name,
        "faithfulness": round(supported / n, 3),
        "citation_rate": round(cited / n, 3),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--judge", action="store_true", help="also score answers with the LLM")
    parser.add_argument("--delay", type=float, default=2.0, help="seconds between LLM calls")
    parser.add_argument("--out", type=Path, default=Path("../eval/results"))
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if not questions:
        raise SystemExit("No questions found.")

    with db.connect() as conn:
        retrieval = evaluate_retrieval(conn, questions, list(Mode), args.k)
        report = {"questions": len(questions), "k": args.k, "retrieval": retrieval}
        print(f"\n{len(questions)} questions, k={args.k}\n")
        print(f"| mode | hit@{args.k} | MRR |\n|---|---|---|")
        for mode, m in retrieval.items():
            print(f"| {mode} | {m['hit_rate']:.0%} | {m['mrr']:.3f} |")
        if args.judge:
            try:
                answers = evaluate_answers(conn, questions, args.k, args.delay)
            except LLMError as exc:
                raise SystemExit(f"LLM error: {exc}") from exc
            report["answers"] = answers
            print(f"\nfaithfulness: {answers['faithfulness']:.0%}"
                  f"   citation rate: {answers['citation_rate']:.0%}   ({answers['model']})")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"eval-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
