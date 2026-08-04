import db
import indexer
from models import Act, Article


def _fake_embed(texts):
    return [[0.1] * 1024 for _ in texts]


def _act(key="labour_code", n_articles=3) -> Act:
    return Act(
        key=key,
        title="Трудовой кодекс Республики Казахстан",
        url=f"https://adilet.zan.kz/rus/docs/{key}",
        doc_type="кодекс",
        act_date="2015-11-23",
        act_number="414-V",
        revision_date="2026-06-11",
        articles=[
            Article(
                number=str(i),
                title=f"Статья про {i}",
                anchor=f"z{i}",
                text=f"Содержание нормы номер {i}. " * 5,
            )
            for i in range(1, n_articles + 1)
        ],
    )


def _conn(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.create_schema(conn)
    return conn


def test_index_act_writes_act_chunks_and_vectors(tmp_path):
    conn = _conn(tmp_path)
    count = indexer.index_act(conn, _act(), embed_fn=_fake_embed)
    assert count == 3
    assert conn.execute("SELECT COUNT(*) FROM acts").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0] == 3


def test_index_act_stores_metadata(tmp_path):
    conn = _conn(tmp_path)
    indexer.index_act(conn, _act(), embed_fn=_fake_embed)
    row = conn.execute(
        "SELECT title, act_date, act_number, revision_date FROM acts"
    ).fetchone()
    assert row[1] == "2015-11-23"
    assert row[2] == "414-V"
    assert row[3] == "2026-06-11"


def test_reindexing_replaces_previous_version(tmp_path):
    """Повторная индексация того же НПА не должна плодить дубликаты."""
    conn = _conn(tmp_path)
    indexer.index_act(conn, _act(n_articles=3), embed_fn=_fake_embed)
    indexer.index_act(conn, _act(n_articles=2), embed_fn=_fake_embed)
    assert conn.execute("SELECT COUNT(*) FROM acts").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0] == 2


def test_fts_index_populated(tmp_path):
    conn = _conn(tmp_path)
    indexer.index_act(conn, _act(), embed_fn=_fake_embed)
    rows = conn.execute(
        "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'нормы'"
    ).fetchone()
    assert rows[0] == 3


def test_embeddings_requested_in_batches(tmp_path):
    conn = _conn(tmp_path)
    batches = []

    def spy(texts):
        batches.append(len(texts))
        return [[0.1] * 1024 for _ in texts]

    indexer.index_act(conn, _act(n_articles=5), embed_fn=spy, batch_size=2)
    assert batches == [2, 2, 1]


def test_index_directory_processes_all_files(tmp_path):
    from scraper import act_to_json

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.json").write_text(act_to_json(_act("a", 2)), encoding="utf-8")
    (raw / "b.json").write_text(act_to_json(_act("b", 3)), encoding="utf-8")
    conn = _conn(tmp_path)
    result = indexer.index_directory(conn, raw, embed_fn=_fake_embed)
    assert result == {"a": 2, "b": 3}
