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
# Кандидатов должно быть заведомо больше, чем чанков в самой длинной статье в
# корпусе, иначе она одна съедает весь пул до дедупликации по статье (статья 16
# ТК РК — 37 чанков). Потолок, который это подразумевает: акт со статьёй из
# 150+ чанков снова столкнётся с этой проблемой.
CANDIDATES = 150
# Доля индексированных чанков, при превышении которой термин запроса считается
# неинформативным для корпуса (не стоп-слово языка, а слово, ставшее вездесущим
# именно в этом корпусе — "статья", "работник", "республики"). FTS может обойти
# порог векторной релевантности, только если хотя бы один термин запроса
# встречается реже этой доли.
DISCRIMINATIVE_DF_FRACTION = 0.10
# Нижняя граница: без стемминга FTS5 считает "кодекс" и "кодекса" разными
# токенами, поэтому редкое юридическое клише вроде "настоящего Кодекса"
# (упоминается лишь в единичных статьях как ссылка на сам акт, а не как
# признак темы запроса) статистически неотличимо от по-настоящему
# специфичного термина. Термин должен встречаться не реже этой доли, иначе
# он слишком разрежен, чтобы быть надёжным сигналом, а не случайным совпадением.
MIN_DISCRIMINATIVE_DF_FRACTION = 0.01
_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Русские стоп-слова: не должны в одиночку "спасать" запрос от отказа —
# они встречаются почти в любом юридическом тексте, поэтому FTS-совпадение
# только по ним не говорит о релевантности.
STOPWORDS: frozenset[str] = frozenset(
    {
        "как", "что", "чего", "чему", "чем", "где", "когда", "куда", "зачем", "почему",
        "кто", "кого", "кому", "это", "этот", "эта", "эти", "тот", "той",
        "какой", "какая", "какое", "какие", "какого", "какому", "каким", "какими", "каком",
        "для", "при", "над", "под", "про", "без", "через", "между",
        "если", "чтобы", "или", "либо", "ибо", "так", "там", "тут",
        "уже", "ещё", "еще",
        "был", "была", "было", "были", "быть", "есть", "нет",
        "все", "всё", "весь", "вся", "свой", "своя", "свои",
        "его", "её", "ее", "их", "они", "оно", "она",
        "мне", "мной", "меня", "вам", "вас", "нам", "нас", "ним", "них", "ему", "ней",
        "будет", "может", "можно", "нужно", "надо",
    }
)

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


def _fts_terms(text: str) -> list[str]:
    """Слова запроса, годные для FTS: не мусор, не стоп-слово языка."""
    words = _WORD_RE.findall(text.lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def _fts_query(terms: list[str]) -> str:
    """Превращает список слов в безопасный запрос FTS5 (экранирует ввод)."""
    return " OR ".join(f'"{w}"' for w in terms)


def _is_discriminative(conn: sqlite3.Connection, terms: list[str]) -> bool:
    """FTS может перебить порог векторной релевантности, только если хотя бы
    один термин запроса информативен для корпуса, а не встречается почти
    везде (как "статья" или "кодекс" — они не стоп-слова языка, но стали
    стоп-словами для этого корпуса)."""
    total = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    if not total:
        return False
    for term in terms:
        df = conn.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?", (f'"{term}"',)
        ).fetchone()[0]
        # df == 0 — термин ничего не нашёл, это не сигнал, а его отсутствие.
        # Слишком низкий df — тоже не сигнал, а шум единичного совпадения.
        if total * MIN_DISCRIMINATIVE_DF_FRACTION < df < total * DISCRIMINATIVE_DF_FRACTION:
            return True
    return False


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


def _fts_candidates(conn: sqlite3.Connection, terms: list[str]) -> list[int]:
    match = _fts_query(terms)
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
    if not query or not query.strip():
        return []

    vector_ids, best_similarity = _vector_candidates(conn, query, embed_fn)
    terms = _fts_terms(query)
    fts_ids = _fts_candidates(conn, terms)

    if best_similarity < SETTINGS.relevance_threshold:
        if not fts_ids or not _is_discriminative(conn, terms):
            return []
    if not vector_ids and not fts_ids:
        return []

    fused = _fuse(vector_ids, fts_ids)
    if not fused:
        return []

    # Строки тянем для всех кандидатов, не только для первых k: длинные статьи
    # дают несколько чанков, и обрезать по k нужно уже после дедупликации по
    # статье — иначе одна статья в нескольких чанках вытесняет остальные нормы.
    placeholders = ",".join("?" * len(fused))
    rows = conn.execute(
        f"""
        SELECT c.id, a.title, c.article_no, c.article_title, c.part_no, c.text,
               a.adilet_url, c.anchor, a.revision_date, a.id
        FROM chunks c JOIN acts a ON a.id = c.act_id
        WHERE c.id IN ({placeholders})
        """,
        [chunk_id for chunk_id, _ in fused],
    ).fetchall()

    by_id = {row[0]: row for row in rows}
    seen_articles: set[tuple[int, str]] = set()
    hits: list[Hit] = []
    for chunk_id, score in fused:
        row = by_id.get(chunk_id)
        if row is None:
            continue
        article_key = (row[9], row[2])
        if article_key in seen_articles:
            continue
        seen_articles.add(article_key)
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
        if len(hits) >= k:
            break
    return hits
