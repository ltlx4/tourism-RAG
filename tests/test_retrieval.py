from pathlib import Path

import pytest

from dubai_rag.ingest import ingest_directory
from dubai_rag.providers import HashEmbedding
from dubai_rag.retrieval import HybridRetriever
from dubai_rag.store import ChunkStore


@pytest.fixture()
def retriever(tmp_path: Path) -> HybridRetriever:
    store = ChunkStore(tmp_path / "test.db")
    embeddings = HashEmbedding()
    corpus = Path(__file__).parents[1] / "data" / "corpus"
    ingest_directory(corpus, store, embeddings)
    return HybridRetriever(store, embeddings, enable_rerank=False, top_k=4)


@pytest.mark.parametrize(
    ("query", "expected_source"),
    [
        ("How can I use the Metro, tram, buses and taxis?", "transport"),
        ("Plan an afternoon in the spice and gold souks", "old-dubai"),
        ("What should a family do with young children?", "families"),
        ("Is dune driving suitable with back problems?", "desert"),
        ("Where can I see yachts and walk around JBR?", "marina"),
        ("I need step-free wheelchair access", "accessible"),
        ("Can I hike and kayak in the mountains?", "hatta"),
    ],
)
def test_retrieval_finds_relevant_source(
    retriever: HybridRetriever, query: str, expected_source: str
):
    results = retriever.search(query)
    assert expected_source in {result.chunk.source_id for result in results[:3]}


def test_results_include_citation_metadata(retriever: HybridRetriever):
    result = retriever.search("Dubai Creek abra")[0].chunk
    assert result.title
    assert result.source_url.startswith("https://")
