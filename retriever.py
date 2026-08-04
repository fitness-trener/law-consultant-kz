"""Гибридный поиск по корпусу НПА.

Векторный поиск проваливается на точных запросах и редких терминах,
полнотекстовый — на переформулировках. Берём оба и сливаем по RRF.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Callable

import llm
from config import load_settings
from db import serialize_vector

SETTINGS = load_settings()
RRF_K = 60
CANDIDATES = 20
_WORD_RE = re.compile(r"\w+", re.UNICODE)

EmbedFn = Callable[[list[str]], list[list[float]]]


@dataclass
class Hit:
    chunk_id: int
    act_title: str
    article_no: str
    article_title: str
    part_no: str | None
    text: str
    url: str
    anchor: str
    revision_date: str | None
    score: float

    @property
    def citation(self) -> str:
        return f"статья {self.article_no} — {self.act_title}"

    @property
    def link(self) -> str:
        return f"{self.url}#{self.anchor}" if self.anchor else self.url


def _fts_query(text: str) -> str:
    """Превращает свободный ввод в безопасный запрос FTS5."""
    words = _WORD_RE.findall(text.lower())
    return " OR ".join(f'"{w}"' for w in words if len(w) > 2)


def _vector_candidates(
    conn: sqlite3.Connection, query: str, embed_fn: EmbedFn
) -> tuple[list[int], float]:
    vector = embed_fn([query])[0]
    rows = conn.execute(
        "SELECT chunk_id, distance FROM vec_chunks"
        " WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (serialize_vector(vector), CANDIDATES),
    ).fetchall()
    if not rows:
        return [], 0.0
    # vec0 отдаёт L2-расстояние по нормализованным векторам: cos = 1 - d²/2
    best_similarity = 1.0 - (rows[0][1] ** 2) / 2.0
    return [row[0] for row in rows], best_similarity


def _fts_candidates(conn: sqlite3.Connection, query: str) -> list[int]:
    match = _fts_query(query)
    if not match:
        return []
    rows = conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?"
        " ORDER BY rank LIMIT ?",
        (match, CANDIDATES),
    ).fetchall()
    return [row[0] for row in rows]


def _fuse(*rankings: list[int]) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + position)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def search(
    conn: sqlite3.Connection,
    query: str,
    k: int = 4,
    embed_fn: EmbedFn = llm.embed,
) -> list[Hit]:
    vector_ids, best_similarity = _vector_candidates(conn, query, embed_fn)
    fts_ids = _fts_candidates(conn, query)

    if best_similarity < SETTINGS.relevance_threshold and not fts_ids:
        return []
    if not vector_ids and not fts_ids:
        return []

    fused = _fuse(vector_ids, fts_ids)[:k]
    if not fused:
        return []

    placeholders = ",".join("?" * len(fused))
    rows = conn.execute(
        f"""
        SELECT c.id, a.title, c.article_no, c.article_title, c.part_no, c.text,
               a.adilet_url, c.anchor, a.revision_date
        FROM chunks c JOIN acts a ON a.id = c.act_id
        WHERE c.id IN ({placeholders})
        """,
        [chunk_id for chunk_id, _ in fused],
    ).fetchall()

    by_id = {row[0]: row for row in rows}
    hits: list[Hit] = []
    for chunk_id, score in fused:
        row = by_id.get(chunk_id)
        if row is None:
            continue
        hits.append(
            Hit(
                chunk_id=row[0],
                act_title=row[1],
                article_no=row[2],
                article_title=row[3],
                part_no=row[4],
                text=row[5],
                url=row[6],
                anchor=row[7],
                revision_date=row[8],
                score=score,
            )
        )
    return hits
