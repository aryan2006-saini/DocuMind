"""Prompting and citation handling for grounded answers."""

import re

from app.retrieval import Hit

NO_ANSWER = "I could not find the answer in the uploaded documents."

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered sources the user gives you.

Rules:
- Every sentence that states a fact must end with the number of its source in square
  brackets, like [1] or [2][3].
- Use only what the sources say. Do not add outside knowledge.
- If the sources do not contain the answer, reply exactly: "{NO_ANSWER}"
- Be concise: at most 6 sentences, or a short bullet list when the question asks for a list.
"""


def build_prompt(question: str, hits: list[Hit]) -> str:
    sources = "\n\n".join(
        f"[{n}] {hit.filename}, page {hit.page}\n{hit.content}" for n, hit in enumerate(hits, 1)
    )
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def parse_citations(answer: str, source_count: int) -> list[int]:
    """Source numbers cited in the answer, in order of first use. Handles [1], [1][2], [1, 2].
    Numbers outside 1..source_count (hallucinated sources) are dropped."""
    seen: list[int] = []
    for group in _CITATION.findall(answer):
        for part in group.split(","):
            number = int(part)
            if 1 <= number <= source_count and number not in seen:
                seen.append(number)
    return seen
