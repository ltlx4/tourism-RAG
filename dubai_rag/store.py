from __future__ import annotations

import json
import re
import sqlite3
from array import array
from pathlib import Path

import numpy as np

from .models import Chunk, SearchResult


class ChunkStore:
    def __init__(self, path: Path):
        self.path = path
        self._semantic_cache: tuple[list[Chunk], np.ndarray] | None = None
        path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    category TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    retrieved_at TEXT NOT NULL DEFAULT '',
                    source_type TEXT NOT NULL DEFAULT 'curated',
                    embedding BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED, title, text, category, tokenize='porter unicode61'
                );
                """
            )

    def replace_all(self, chunks: list[Chunk]) -> None:
        self.initialize()
        with self.connect() as db:
            columns = {row[1] for row in db.execute("PRAGMA table_info(chunks)")}
            if "retrieved_at" not in columns:
                db.execute("ALTER TABLE chunks ADD COLUMN retrieved_at TEXT NOT NULL DEFAULT ''")
                db.execute(
                    "ALTER TABLE chunks ADD COLUMN source_type TEXT NOT NULL DEFAULT 'curated'"
                )
            db.execute("DELETE FROM chunks")
            db.execute("DELETE FROM chunks_fts")
            db.executemany(
                """
                INSERT INTO chunks
                (chunk_id, source_id, title, text, source_url, category, ordinal,
                 retrieved_at, source_type, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.chunk_id,
                        c.source_id,
                        c.title,
                        c.text,
                        c.source_url,
                        c.category,
                        c.ordinal,
                        c.retrieved_at,
                        c.source_type,
                        array("f", c.embedding).tobytes(),
                    )
                    for c in chunks
                ],
            )
            db.executemany(
                "INSERT INTO chunks_fts (chunk_id, title, text, category) VALUES (?, ?, ?, ?)",
                [(c.chunk_id, c.title, c.text, c.category) for c in chunks],
            )
        self._semantic_cache = None

    def set_metadata(self, key: str, value: str) -> None:
        self.initialize()
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", (key, value)
            )

    def metadata(self) -> dict[str, str]:
        self.initialize()
        with self.connect() as db:
            return {row["key"]: row["value"] for row in db.execute("SELECT * FROM metadata")}

    def count(self) -> int:
        self.initialize()
        with self.connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        stored = row["embedding"]
        if isinstance(stored, str):
            embedding = json.loads(stored)
        else:
            values = array("f")
            values.frombytes(stored)
            embedding = list(values)
        return Chunk(
            chunk_id=row["chunk_id"],
            source_id=row["source_id"],
            title=row["title"],
            text=row["text"],
            source_url=row["source_url"],
            category=row["category"],
            ordinal=row["ordinal"],
            retrieved_at=row["retrieved_at"] if "retrieved_at" in row.keys() else "",
            source_type=row["source_type"] if "source_type" in row.keys() else "curated",
            embedding=embedding,
        )

    def lexical_search(self, query: str, limit: int) -> list[SearchResult]:
        terms = re.findall(r"[\w']+", query)
        if not terms:
            return []
        fts_query = " OR ".join(f'"{term}"' for term in terms[:20])
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT c.*, bm25(chunks_fts, 0, 4, 1, 2) AS rank_score
                FROM chunks_fts JOIN chunks c USING (chunk_id)
                WHERE chunks_fts MATCH ?
                ORDER BY rank_score LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        return [
            SearchResult(chunk=self._row_to_chunk(row), score=-row["rank_score"])
            for row in rows
        ]

    def semantic_search(self, query_vector: list[float], limit: int) -> list[SearchResult]:
        if self._semantic_cache is None:
            with self.connect() as db:
                chunks = [self._row_to_chunk(row) for row in db.execute("SELECT * FROM chunks")]
            matrix = np.asarray([chunk.embedding for chunk in chunks], dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.maximum(norms, 1e-12)
            self._semantic_cache = chunks, matrix
        chunks, matrix = self._semantic_cache
        query = np.asarray(query_vector, dtype=np.float32)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        scores = matrix @ query
        indices = np.argsort(scores)[::-1][:limit]
        return [
            SearchResult(chunk=chunks[int(index)], score=float(scores[index]))
            for index in indices
        ]
