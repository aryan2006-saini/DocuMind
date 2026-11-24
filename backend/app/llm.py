"""LLM clients. Groq exposes an OpenAI-compatible chat API, so plain httpx is enough."""

import re
from typing import Protocol

import httpx

from app.config import get_settings

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class LLMError(Exception):
    """The LLM could not answer (missing key, network error, rate limit...)."""


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class GroqLLM:
    def __init__(self, api_key: str, model: str, timeout_s: float):
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set. Add it to the .env file.")
        self.api_key = api_key
        self.model = model
        self.name = f"groq:{model}"
        self.timeout_s = timeout_s

    def complete(self, system: str, user: str) -> str:
        try:
            response = httpx.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "temperature": 0.1,
                    "max_tokens": 800,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=self.timeout_s,
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"Could not reach Groq: {exc}") from exc
        if response.status_code == 429:
            raise LLMError("Groq rate limit reached. Wait a minute and try again.")
        if response.status_code != 200:
            raise LLMError(f"Groq returned {response.status_code}: {response.text[:300]}")
        return response.json()["choices"][0]["message"]["content"].strip()


class FakeLLM:
    """Offline stand-in: answers with the first sentence of the first source and cites it."""

    name = "fake"

    def complete(self, system: str, user: str) -> str:
        match = re.search(r"\[1\][^\n]*\n(.+?)(?:\n\[2\]|\Z)", user, re.DOTALL)
        if not match:
            return "I could not find the answer in the uploaded documents."
        first_sentence = re.split(r"(?<=[.!?])\s", match.group(1).strip(), maxsplit=1)[0]
        return f"{first_sentence} [1]"


def get_llm() -> LLM:
    settings = get_settings()
    if settings.llm_provider == "fake":
        return FakeLLM()
    return GroqLLM(settings.groq_api_key, settings.groq_model, settings.llm_timeout_s)
