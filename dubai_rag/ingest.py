from __future__ import annotations

import re
from pathlib import Path

from .chunking import chunk_document
from .models import Document
from .providers import EmbeddingProvider
from .store import ChunkStore


def _parse_markdown(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    metadata: dict[str, str] = {}
    body = raw
    if raw.startswith("---\n"):
        _, header, body = raw.split("---\n", 2)
        for line in header.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()
    title_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    title = metadata.get("title") or (title_match.group(1) if title_match else path.stem)
    return Document(
        source_id=path.stem,
        title=title,
        text=body.strip(),
        source_url=metadata.get("source_url", ""),
        category=metadata.get("category", "general"),
    )


def ingest_directory(path: Path, store: ChunkStore, embeddings: EmbeddingProvider) -> int:
    documents = [_parse_markdown(file) for file in sorted(path.glob("*.md"))]
    chunks = [chunk for document in documents for chunk in chunk_document(document)]
    vectors = embeddings.embed([f"{chunk.title}\n{chunk.text}" for chunk in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk.embedding = vector
    store.replace_all(chunks)
    return len(chunks)

