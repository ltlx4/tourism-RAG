from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import Settings
from .ingest import ingest_directory
from .providers import ProviderError, build_embeddings, build_llm
from .retrieval import HybridRetriever
from .service import RAGService
from .store import ChunkStore


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    history: list[Message] = Field(default_factory=list, max_length=12)


@lru_cache
def get_components() -> tuple[Settings, ChunkStore, RAGService]:
    settings = Settings.from_env()
    store = ChunkStore(settings.database_path)
    embeddings = build_embeddings(settings)
    llm = build_llm(settings)
    if store.count() == 0:
        ingest_directory(settings.corpus_path, store, embeddings)
    retriever = HybridRetriever(
        store,
        embeddings,
        llm,
        settings.candidate_k,
        settings.top_k,
        settings.enable_rerank,
    )
    return settings, store, RAGService(settings, retriever, llm)


app = FastAPI(title="DXB Guide RAG", version="0.1.0")


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(Path(__file__).with_name("static").joinpath("index.html").read_text())


@app.get("/api/health")
def health() -> dict[str, object]:
    settings, store, _ = get_components()
    return {
        "status": "ok",
        "chunks": store.count(),
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embedding_provider": settings.embedding_provider,
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, object]:
    _, _, service = get_components()
    try:
        return service.answer(
            request.question, [message.model_dump() for message in request.history]
        )
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

