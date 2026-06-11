from __future__ import annotations

import hashlib
import re

from .models import Chunk, Document


def chunk_document(document: Document, target_words: int = 110, overlap_words: int = 20) -> list[Chunk]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", document.text) if p.strip()]
    groups: list[str] = []
    current: list[str] = []
    count = 0

    for paragraph in paragraphs:
        words = paragraph.split()
        if current and count + len(words) > target_words:
            groups.append("\n\n".join(current))
            overlap = " ".join(" ".join(current).split()[-overlap_words:])
            current = [overlap, paragraph] if overlap else [paragraph]
            count = len(current[0].split()) + len(words)
        else:
            current.append(paragraph)
            count += len(words)
    if current:
        groups.append("\n\n".join(current))

    chunks = []
    for ordinal, text in enumerate(groups):
        digest = hashlib.sha1(
            f"{document.source_id}:{ordinal}:{text}".encode(), usedforsecurity=False
        ).hexdigest()[:16]
        chunks.append(
            Chunk(
                chunk_id=digest,
                source_id=document.source_id,
                title=document.title,
                text=text,
                source_url=document.source_url,
                category=document.category,
                ordinal=ordinal,
                retrieved_at=document.retrieved_at,
                source_type=document.source_type,
            )
        )
    return chunks
