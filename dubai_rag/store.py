from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path

from .models import Chunk, SearchResult


class ChunkStore:
    def __init__(self, path: Path):
        self.path = path
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
                    embedding TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED, title, text, category, tokenize='porter unicode61'
                );
                """
            )

    def replace_all(self, chunks: list[Chunk]) -> None:
        self.initialize()
        with self.connect() as db:
            db.execute("DELETE FROM chunks")
            db.execute("DELETE FROM chunks_fts")
            db.executemany(
                """
                INSERT INTO chunks
                (chunk_id, source_id, title, text, source_url, category, ordinal, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
                        json.dumps(c.embedding),
                    )
                    for c in chunks
                ],
            )
            db.executemany(
                "INSERT INTO chunks_fts (chunk_id, title, text, category) VALUES (?, ?, ?, ?)",
                [(c.chunk_id, c.title, c.text, c.category) for c in chunks],
            )

    def count(self) -> int:
        self.initialize()
        with self.connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"],
            source_id=row["source_id"],
            title=row["title"],
            text=row["text"],
            source_url=row["source_url"],
            category=row["category"],
            ordinal=row["ordinal"],
            embedding=json.loads(row["embedding"]),
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
        with self.connect() as db:
            rows = db.execute("SELECT * FROM chunks").fetchall()
        scored = []
        for row in rows:
            chunk = self._row_to_chunk(row)
            score = sum(a * b for a, b in zip(query_vector, chunk.embedding))
            query_norm = math.sqrt(sum(a * a for a in query_vector)) or 1.0
            chunk_norm = math.sqrt(sum(b * b for b in chunk.embedding)) or 1.0
            scored.append(SearchResult(chunk=chunk, score=score / (query_norm * chunk_norm)))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]

