from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .config import Settings
from .providers import build_embeddings, build_llm
from .retrieval import HybridRetriever
from .store import ChunkStore


@dataclass(frozen=True)
class EvalCase:
    query: str
    expected_sources: tuple[str, ...]


def load_cases(path: Path) -> list[EvalCase]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            cases.append(EvalCase(item["query"], tuple(item["expected_sources"])))
    return cases


def evaluate(retriever: HybridRetriever, cases: list[EvalCase]) -> dict[str, float]:
    recall_at_1 = 0
    recall_at_3 = 0
    reciprocal_rank = 0.0
    durations = []
    for case in cases:
        started = perf_counter()
        results = retriever.search(case.query)
        durations.append((perf_counter() - started) * 1000)
        sources = [result.chunk.source_id for result in results]
        relevant_ranks = [
            rank for rank, source in enumerate(sources, 1) if source in case.expected_sources
        ]
        if relevant_ranks:
            best_rank = min(relevant_ranks)
            recall_at_1 += best_rank <= 1
            recall_at_3 += best_rank <= 3
            reciprocal_rank += 1 / best_rank
    count = len(cases) or 1
    return {
        "cases": float(len(cases)),
        "recall_at_1": round(recall_at_1 / count, 4),
        "recall_at_3": round(recall_at_3 / count, 4),
        "mrr": round(reciprocal_rank / count, 4),
        "mean_latency_ms": round(sum(durations) / count, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark retrieval quality")
    parser.add_argument("--dataset", type=Path, default=Path("data/evaluation/retrieval.jsonl"))
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_env()
    embeddings = build_embeddings(settings)
    llm = build_llm(settings) if args.rerank else None
    retriever = HybridRetriever(
        ChunkStore(settings.database_path),
        embeddings,
        llm,
        settings.candidate_k,
        settings.top_k,
        args.rerank,
    )
    print(json.dumps(evaluate(retriever, load_cases(args.dataset)), indent=2))


if __name__ == "__main__":
    main()

