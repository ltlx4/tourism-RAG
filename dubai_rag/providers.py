from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from .config import Settings


class ProviderError(RuntimeError):
    pass


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ProviderError(f"Provider request failed at {url}: {exc}") from exc


class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        raise NotImplementedError


class OllamaLLM(LLMProvider):
    def __init__(self, base_url: str, model: str, max_tokens: int = 450):
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        data = _post_json(
            f"{self.base_url}/api/chat",
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "think": False,
                "options": {"temperature": temperature, "num_predict": self.max_tokens},
            },
        )
        return data["message"]["content"].strip()


class OpenAICompatibleLLM(LLMProvider):
    def __init__(self, base_url: str, model: str, api_key: str, max_tokens: int = 450):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        data = _post_json(
            f"{self.base_url}/v1/chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": self.max_tokens,
            },
            {"Authorization": f"Bearer {self.api_key}"},
        )
        return data["choices"][0]["message"]["content"].strip()


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class HashEmbedding(EmbeddingProvider):
    """Deterministic feature hashing for a zero-download local baseline."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    @property
    def name(self) -> str:
        return f"hash-{self.dimensions}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        normalized = re.sub(r"\s+", " ", text.lower()).strip()
        tokens = re.findall(r"[\w']+", normalized)
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log1p(len(feature)))
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class OllamaEmbedding(EmbeddingProvider):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = _post_json(
            f"{self.base_url}/api/embed",
            {"model": self.model, "input": texts},
        )
        return data["embeddings"]


class OpenAIEmbedding(EmbeddingProvider):
    def __init__(self, base_url: str, model: str, api_key: str):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = _post_json(
            f"{self.base_url}/v1/embeddings",
            {"model": self.model, "input": texts},
            {"Authorization": f"Bearer {self.api_key}"},
        )
        return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]


def build_llm(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "ollama":
        return OllamaLLM(settings.llm_base_url, settings.llm_model, settings.max_tokens)
    if settings.llm_provider in {"openai", "openai-compatible"}:
        return OpenAICompatibleLLM(
            settings.llm_base_url,
            settings.llm_model,
            settings.llm_api_key,
            settings.max_tokens,
        )
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def build_embeddings(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "hash":
        dimensions = int(settings.embedding_model.rsplit("-", 1)[-1])
        return HashEmbedding(dimensions)
    if settings.embedding_provider == "ollama":
        return OllamaEmbedding(settings.embedding_base_url, settings.embedding_model)
    if settings.embedding_provider in {"openai", "openai-compatible"}:
        return OpenAIEmbedding(
            settings.embedding_base_url,
            settings.embedding_model,
            settings.embedding_api_key,
        )
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
