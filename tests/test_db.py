import sqlite3

import pytest

import db


def test_connect_loads_sqlite_vec(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    version = conn.execute("SELECT vec_version()").fetchone()[0]
    assert version.startswith("v0.1")


def test_create_schema_creates_all_tables(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.create_schema(conn)
    names = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"acts", "chunks", "vec_chunks", "chunks_fts"} <= names


def test_create_schema_is_idempotent(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.create_schema(conn)
    db.create_schema(conn)  # не должно бросать


def test_vector_roundtrip(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.create_schema(conn)
    conn.execute(
        "INSERT INTO acts (id, title, adilet_url, doc_type, revision_date, scraped_at)"
        " VALUES (1, 'ТК РК', 'https://x', 'кодекс', '2026-07-01', '2026-08-04')"
    )
    conn.execute(
        "INSERT INTO chunks (id, act_id, article_no, article_title, part_no, text,"
        " char_len, anchor) VALUES (1, 1, '1', 'Понятия', NULL, 'текст', 5, 'z1')"
    )
    conn.execute(
        "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
        (1, db.serialize_vector([0.5] * 1024)),
    )
    conn.commit()
    row = conn.execute(
        "SELECT chunk_id, distance FROM vec_chunks"
        " WHERE embedding MATCH ? AND k = 1",
        (db.serialize_vector([0.5] * 1024),),
    ).fetchone()
    assert row[0] == 1
    assert row[1] == pytest.approx(0.0, abs=1e-4)


def test_fts_finds_chunk_by_word(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.create_schema(conn)
    conn.execute(
        "INSERT INTO acts (id, title, adilet_url, doc_type, revision_date, scraped_at)"
        " VALUES (1, 'ТК РК', 'https://x', 'кодекс', '2026-07-01', '2026-08-04')"
    )
    conn.execute(
        "INSERT INTO chunks (id, act_id, article_no, article_title, part_no, text,"
        " char_len, anchor) VALUES (1, 1, '30', 'Договор', NULL,"
        " 'Трудовой договор заключается в письменной форме', 46, 'z30')"
    )
    conn.commit()
    rows = conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'договор'"
    ).fetchall()
    assert [r[0] for r in rows] == [1]


def test_foreign_key_enforced(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.create_schema(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO chunks (act_id, article_no, article_title, part_no, text,"
            " char_len, anchor) VALUES (999, '1', 'X', NULL, 'y', 1, 'z1')"
        )


def test_corpus_summary_lists_acts(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.create_schema(conn)
    conn.execute(
        "INSERT INTO acts (title, adilet_url, doc_type, revision_date, scraped_at)"
        " VALUES ('ТК РК', 'https://x', 'кодекс', '2026-07-01', '2026-08-04')"
    )
    conn.commit()
    assert db.corpus_summary(conn) == [("ТК РК", "2026-07-01")]
