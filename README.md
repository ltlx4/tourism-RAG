# DXB Guide: Dubai Tourism RAG

An Ollama-first retrieval-augmented generation system that turns an officially sourced
Dubai travel corpus into grounded, cited answers. Retrieval is measurable and inspectable,
providers are swappable, and the complete default path runs locally.

## What makes it useful

- Hybrid retrieval combines SQLite FTS5 with real `nomic-embed-text` semantic vectors.
- Source-aware reciprocal rank fusion combines rank, score strength and result diversity.
- NumPy-vectorized similarity search caches normalized vectors instead of parsing every
  embedding on every request.
- Answers carry numbered citations linked to official tourism sources; invalid citation
  IDs are removed and sentence-level coverage is reported.
- Conversation follow-ups are contextualized without an extra model call by default;
  optional LLM query rewriting remains configurable.
- Prompt-injection boundaries treat retrieved text as data, not instructions.
- Ollama, OpenAI, and OpenAI-compatible endpoints share the same provider interfaces.
- Index metadata prevents querying vectors with the wrong embedding model.
- Request IDs and retrieval/generation timings expose runtime behavior.
- A versioned 16-question benchmark reports Recall@K, MRR and retrieval latency.
- The index is local SQLite, so no vector database account is required.

## Measured retrieval quality

Default local configuration on the included 35-chunk, 16-topic corpus:

| Metric | Result |
| --- | ---: |
| Recall@1 | 1.000 |
| Recall@3 | 1.000 |
| MRR | 1.000 |
| Mean retrieval latency | 40.2 ms |

These are small-corpus regression metrics, not a claim of general search quality. Run the
benchmark after every corpus, chunking or embedding change:

```bash
dubai-rag-eval
```

## Architecture

```text
Markdown corpus
    -> paragraph-aware chunking
    -> FTS5 index + Nomic embeddings + provenance metadata

Question
    -> zero-call conversation-aware query
    -> lexical search + semantic search
    -> source-aware rank fusion + diversity
    -> optional Qwen reranking (off by default)
    -> grounded Qwen answer
    -> citation validation + source cards + timings
```

## Run locally with Qwen

Prerequisites: Python 3.11+ and Ollama.

```bash
ollama pull qwen3.5:4b
ollama pull nomic-embed-text
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
python -m dubai_rag.cli ingest
uvicorn dubai_rag.api:app --reload
```

Open `http://127.0.0.1:8000`. You can also query from the terminal:

```bash
python -m dubai_rag.cli ask "Plan a culture-focused day in Old Dubai"
```

The default is real local semantic retrieval with `nomic-embed-text`. A feature-hash
fallback remains available for offline tests or constrained environments:

```bash
RAG_EMBEDDING_PROVIDER=hash
RAG_EMBEDDING_MODEL=hash-384
python -m dubai_rag.cli ingest
```

Rebuild the index whenever the corpus or embedding model changes.

`qwen3.5:4b` is the default answer model for responsive local use; `qwen3.5:9b` remains a
drop-in quality option through `RAG_LLM_MODEL`. LLM reranking is disabled by default because
it roughly doubles model work on local hardware. Set `RAG_ENABLE_RERANK=true` when the
quality/latency tradeoff is appropriate.

## Use a paid or hosted API

Any endpoint implementing OpenAI's chat and embedding routes can be configured:

```bash
RAG_LLM_PROVIDER=openai
RAG_LLM_MODEL=gpt-4.1-mini
RAG_LLM_BASE_URL=https://api.openai.com
RAG_LLM_API_KEY=...
RAG_EMBEDDING_PROVIDER=openai
RAG_EMBEDDING_MODEL=text-embedding-3-small
RAG_EMBEDDING_BASE_URL=https://api.openai.com
RAG_EMBEDDING_API_KEY=...
```

Then run ingestion again so stored and query vectors use the same model.

## API

`POST /api/chat`

```json
{
  "question": "What is the easiest way to see the Creek and souks?",
  "history": []
}
```

The response includes `answer`, `sources`, the retrieval `query`, citation diagnostics and
per-stage timings.
`GET /api/health` reports index and provider status without invoking the LLM.

## Tests

```bash
ruff check .
pytest
```

The 13-test suite covers chunking, retrieval, source metadata, citation validation,
evaluation math and prompt boundaries. CI uses the deterministic hash fallback and does
not require Ollama.

## Source provenance

The repository contains 16 curated topic summaries linked to official Dubai tourism
pages. Each new summary records `source_type` and retrieval date. To attempt timestamped,
gitignored audit snapshots and produce a per-source sync report:

```bash
dubai-rag sync-sources
```

Some official sites reject automated clients; those responses are recorded as failures
instead of aborting the sync. Successful snapshots are intentionally not ingested without
review because tourism pages contain navigation, campaign and volatile booking text that
should not silently enter a trusted corpus.

## Data policy

The bundled corpus emphasizes stable planning knowledge and deliberately avoids hard-coded
prices and opening hours. It is still a compact demonstration corpus, not a complete Dubai
travel authority. Travelers should verify changing schedules, entry rules and venue
policies at the linked source.
