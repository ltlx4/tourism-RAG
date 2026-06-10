from __future__ import annotations

import argparse
import json

from .config import Settings
from .ingest import ingest_directory
from .providers import build_embeddings, build_llm
from .retrieval import HybridRetriever
from .service import RAGService
from .store import ChunkStore


def main() -> None:
    parser = argparse.ArgumentParser(prog="dubai-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="Rebuild the local search index")
    ask = subparsers.add_parser("ask", help="Ask a question from the terminal")
    ask.add_argument("question")
    args = parser.parse_args()

    settings = Settings.from_env()
    store = ChunkStore(settings.database_path)
    embeddings = build_embeddings(settings)
    if args.command == "ingest":
        count = ingest_directory(settings.corpus_path, store, embeddings)
        print(f"Indexed {count} chunks in {settings.database_path}")
        return

    llm = build_llm(settings)
    retriever = HybridRetriever(
        store, embeddings, llm, settings.candidate_k, settings.top_k, settings.enable_rerank
    )
    response = RAGService(settings, retriever, llm).answer(args.question)
    print(response["answer"])
    print("\nSources:")
    print(json.dumps(response["sources"], indent=2))


if __name__ == "__main__":
    main()

