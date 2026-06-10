# DXB Guide: Dubai Tourism RAG

An Ollama-first retrieval-augmented generation system that turns a curated Dubai travel
guide into grounded, cited answers. It is designed as a portfolio project rather than a
thin chatbot wrapper: retrieval remains inspectable, providers are swappable, and the
local path does not require a paid API or a second model download.

## What makes it useful

- Hybrid retrieval combines SQLite FTS5/BM25-style ranking with dense feature-hash vectors.
- Reciprocal rank fusion prevents either retrieval method from dominating.
- `qwen3.5:9b` optionally reranks candidates before generating an answer.
- Answers carry numbered citations linked to official tourism sources.
- Conversation follow-ups are rewritten into standalone retrieval queries.
- Prompt-injection boundaries treat retrieved text as data, not instructions.
- Ollama, OpenAI, and OpenAI-compatible endpoints share the same provider interfaces.
- The index is local SQLite: no vector database account or background service is needed.

## Architecture

```text
Markdown corpus
    -> paragraph-aware chunking
    -> FTS5 index + configurable embeddings

Question
    -> conversation-aware query
    -> lexical search + semantic search
    -> reciprocal rank fusion
    -> optional Qwen reranking
    -> grounded Qwen answer + source cards
```

## Run locally with Qwen

Prerequisites: Python 3.11+ and Ollama with `qwen3.5:9b` installed.

```bash
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

The default `hash-384` embedding provider is deterministic and requires no download.
For stronger semantic retrieval while staying local:

```bash
ollama pull nomic-embed-text
RAG_EMBEDDING_PROVIDER=ollama
RAG_EMBEDDING_MODEL=nomic-embed-text
python -m dubai_rag.cli ingest
```

Rebuild the index whenever the corpus or embedding model changes.

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

The response includes `answer`, `sources`, and the standalone retrieval `query`.
`GET /api/health` reports index and provider status without invoking the LLM.

## Tests

```bash
ruff check .
pytest
```

Retrieval tests assert that benchmark questions surface the expected source. They run
fully offline and do not require Ollama.

## Data policy

The bundled corpus emphasizes stable planning knowledge and links to Dubai's official
tourism portal. It deliberately avoids hard-coded prices and opening hours. Travelers
should verify changing schedules, entry rules and venue policies at the linked source.

