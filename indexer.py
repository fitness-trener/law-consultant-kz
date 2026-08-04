"""Сборка индекса: JSON-акты → чанки → эмбеддинги → база."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import llm
from chunker import chunk_act
from db import serialize_vector
from models import Act
from scraper import act_from_json

EmbedFn = Callable[[list[str]], list[list[float]]]


def index_act(
    conn: sqlite3.Connection,
    act: Act,
    embed_fn: EmbedFn = llm.embed,
    batch_size: int = 16,
) -> int:
    chunks = chunk_act(act)

    # Переиндексация заменяет предыдущую версию НПА целиком. Удаление
    # chunks каскадом (ON DELETE CASCADE) не нужно здесь — триггеры на
    # chunks сами чистят chunks_fts и vec_chunks при удалении строк chunks.
    old = conn.execute("SELECT id FROM acts WHERE adilet_url = ?", (act.url,)).fetchone()
    if old:
        conn.execute("DELETE FROM chunks WHERE act_id = ?", (old[0],))
        conn.execute("DELETE FROM acts WHERE id = ?", (old[0],))

    cursor = conn.execute(
        "INSERT INTO acts (title, adilet_url, doc_type, act_date, act_number,"
        " revision_date, scraped_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            act.title,
            act.url,
            act.doc_type,
            act.act_date,
            act.act_number,
            act.revision_date,
            datetime.now(timezone.utc).date().isoformat(),
        ),
    )
    act_id = cursor.lastrowid

    chunk_ids: list[int] = []
    for chunk in chunks:
        cursor = conn.execute(
            "INSERT INTO chunks (act_id, article_no, article_title, part_no, anchor,"
            " text, char_len) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                act_id,
                chunk.article_no,
                chunk.article_title,
                chunk.part_no,
                chunk.anchor,
                chunk.text,
                chunk.char_len,
            ),
        )
        chunk_ids.append(cursor.lastrowid)

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embed_fn([c.text for c in batch])
        conn.executemany(
            "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
            [
                (chunk_id, serialize_vector(vector))
                for chunk_id, vector in zip(chunk_ids[start : start + batch_size], vectors)
            ],
        )

    conn.commit()
    return len(chunks)


def index_directory(
    conn: sqlite3.Connection, raw_dir: Path, embed_fn: EmbedFn = llm.embed
) -> dict[str, int]:
    result: dict[str, int] = {}
    for path in sorted(raw_dir.glob("*.json")):
        act = act_from_json(path.read_text(encoding="utf-8"))
        result[act.key] = index_act(conn, act, embed_fn=embed_fn)
    return result
