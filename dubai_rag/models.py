from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    source_id: str
    title: str
    text: str
    source_url: str = ""
    category: str = "general"


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    title: str
    text: str
    source_url: str
    category: str
    ordinal: int
    embedding: list[float] = field(default_factory=list)


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    lexical_rank: int | None = None
    semantic_rank: int | None = None
    rerank_score: float | None = None

