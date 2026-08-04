"""Подключение к SQLite и схема базы."""
from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import sqlite_vec

from config import load_settings

SETTINGS = load_settings()

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS acts (
    id            INTEGER PRIMARY KEY,
    title         TEXT NOT NULL,
    adilet_url    TEXT NOT NULL UNIQUE,
    doc_type      TEXT NOT NULL,
    act_date      TEXT,
    act_number    TEXT,
    revision_date TEXT,
    scraped_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY,
    act_id        INTEGER NOT NULL REFERENCES acts(id) ON DELETE CASCADE,
    article_no    TEXT,
    article_title TEXT,
    part_no       TEXT,
    anchor        TEXT,
    text          TEXT NOT NULL,
    char_len      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_act ON chunks(act_id);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding FLOAT[{SETTINGS.embed_dim}]
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 0'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad_vec AFTER DELETE ON chunks BEGIN
    DELETE FROM vec_chunks WHERE chunk_id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path or SETTINGS.db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def serialize_vector(values: list[float]) -> bytes:
    """Упаковывает вектор в формат, который понимает vec0."""
    return struct.pack(f"{len(values)}f", *values)


def corpus_summary(conn: sqlite3.Connection) -> list[tuple[str, str | None]]:
    return [
        (row[0], row[1])
        for row in conn.execute(
            "SELECT title, revision_date FROM acts ORDER BY title"
        )
    ]
