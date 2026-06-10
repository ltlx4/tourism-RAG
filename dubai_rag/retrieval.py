from __future__ import annotations

import json
import re

from .models import SearchResult
from .providers import EmbeddingProvider, LLMProvider, ProviderError
from .store import ChunkStore


class HybridRetriever:
    def __init__(
        self,
        store: ChunkStore,
        embeddings: EmbeddingProvider,
        llm: LLMProvider | None = None,
        candidate_k: int = 18,
        top_k: int = 6,
        enable_rerank: bool = True,
    ):
        self.store = store
        self.embeddings = embeddings
        self.llm = llm
        self.candidate_k = candidate_k
        self.top_k = top_k
        self.enable_rerank = enable_rerank

    def search(self, query: str) -> list[SearchResult]:
        lexical = self.store.lexical_search(query, self.candidate_k)
        vector = self.embeddings.embed([query])[0]
        semantic = self.store.semantic_search(vector, self.candidate_k)

        fused: dict[str, SearchResult] = {}
        for rank, result in enumerate(lexical, 1):
            item = fused.setdefault(
                result.chunk.chunk_id, SearchResult(chunk=result.chunk, score=0.0)
            )
            item.score += 1 / (60 + rank)
            item.lexical_rank = rank
        for rank, result in enumerate(semantic, 1):
            item = fused.setdefault(
                result.chunk.chunk_id, SearchResult(chunk=result.chunk, score=0.0)
            )
            item.score += 1 / (60 + rank)
            item.semantic_rank = rank

        candidates = sorted(fused.values(), key=lambda item: item.score, reverse=True)[
            : self.candidate_k
        ]
        if self.enable_rerank and self.llm and candidates:
            candidates = self._rerank(query, candidates)
        return candidates[: self.top_k]

    def _rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        snippets = [
            {"id": result.chunk.chunk_id, "text": result.chunk.text[:700]}
            for result in candidates[:12]
        ]
        prompt = (
            "Score each passage's relevance to the travel question from 0 to 10. "
            "Return only a JSON object mapping passage id to numeric score.\n"
            f"Question: {query}\nPassages: {json.dumps(snippets, ensure_ascii=False)}"
        )
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}], temperature=0)
            match = re.search(r"\{.*\}", response, re.DOTALL)
            scores = json.loads(match.group(0)) if match else {}
            for result in candidates:
                result.rerank_score = float(scores.get(result.chunk.chunk_id, 0))
            return sorted(
                candidates,
                key=lambda item: (item.rerank_score or 0, item.score),
                reverse=True,
            )
        except (ProviderError, ValueError, TypeError, json.JSONDecodeError):
            return candidates

