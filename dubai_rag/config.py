from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_model: str
    llm_base_url: str
    llm_api_key: str
    embedding_provider: str
    embedding_model: str
    embedding_base_url: str
    embedding_api_key: str
    database_path: Path
    corpus_path: Path
    top_k: int
    candidate_k: int
    enable_rerank: bool
    enable_query_rewrite: bool
    temperature: float
    max_tokens: int
    max_history: int

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        return cls(
            llm_provider=os.getenv("RAG_LLM_PROVIDER", "ollama"),
            llm_model=os.getenv("RAG_LLM_MODEL", "qwen3.5:4b"),
            llm_base_url=os.getenv("RAG_LLM_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
            llm_api_key=os.getenv("RAG_LLM_API_KEY", ""),
            embedding_provider=os.getenv("RAG_EMBEDDING_PROVIDER", "ollama"),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "nomic-embed-text"),
            embedding_base_url=os.getenv(
                "RAG_EMBEDDING_BASE_URL", "http://127.0.0.1:11434"
            ).rstrip("/"),
            embedding_api_key=os.getenv("RAG_EMBEDDING_API_KEY", ""),
            database_path=Path(os.getenv("RAG_DATABASE_PATH", "data/index/dubai_rag.db")),
            corpus_path=Path(os.getenv("RAG_CORPUS_PATH", "data/corpus")),
            top_k=int(os.getenv("RAG_TOP_K", "4")),
            candidate_k=int(os.getenv("RAG_CANDIDATE_K", "18")),
            enable_rerank=_bool("RAG_ENABLE_RERANK", False),
            enable_query_rewrite=_bool("RAG_ENABLE_QUERY_REWRITE", False),
            temperature=float(os.getenv("RAG_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("RAG_MAX_TOKENS", "450")),
            max_history=int(os.getenv("RAG_MAX_HISTORY", "6")),
        )
