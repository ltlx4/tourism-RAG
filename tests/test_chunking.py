from dubai_rag.chunking import chunk_document
from dubai_rag.models import Document


def test_chunking_preserves_metadata_and_overlap():
    document = Document(
        source_id="test",
        title="Test Guide",
        text="\n\n".join(["one two three four five"] * 4),
        source_url="https://example.com",
        category="test",
    )
    chunks = chunk_document(document, target_words=8, overlap_words=2)

    assert len(chunks) > 1
    assert chunks[0].title == "Test Guide"
    assert chunks[0].source_url == "https://example.com"
    assert chunks[0].chunk_id != chunks[1].chunk_id

