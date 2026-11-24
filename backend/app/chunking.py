"""Turn PDF pages into overlapping text chunks that keep their page number."""

import io
import re
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError


@dataclass(frozen=True)
class Chunk:
    page: int  # 1-based page number
    text: str


def extract_pages(data: bytes) -> list[str]:
    """Return the text of each page. Raises ValueError for a file pypdf cannot read."""
    try:
        reader = PdfReader(io.BytesIO(data))
        return [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Could not read the PDF: {exc}") from exc


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"-\n(\w)", r"\1", text)  # re-join words hyphenated across lines
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Break points we prefer, best first: paragraph, sentence, line, word.
_SEPARATORS = ("\n\n", ". ", "? ", "! ", "\n", " ")


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into pieces of at most `size` characters, overlapping by about `overlap`.

    Each cut is moved back to the nearest paragraph, sentence or word boundary found in the
    last 40% of the window, so chunks rarely end mid-sentence.
    """
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= size:
        return [text]

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            for sep in _SEPARATORS:
                cut = window.rfind(sep, int(size * 0.6))
                if cut != -1:
                    end = start + cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        space = text.find(" ", next_start, end)  # start the overlap on a word boundary
        start = space + 1 if space != -1 else next_start
    return pieces


def chunk_pages(pages: list[str], size: int, overlap: int) -> list[Chunk]:
    return [
        Chunk(page=number, text=piece)
        for number, page_text in enumerate(pages, start=1)
        for piece in split_text(page_text, size, overlap)
    ]
