import cli
import db
import indexer
from config import Source
from models import Act, Article


def _fake_embed(texts):
    return [[0.1] * 1024 for _ in texts]


def _act(key: str, url: str) -> Act:
    return Act(
        key=key,
        title=f"Акт «{key}»",
        url=url,
        doc_type="кодекс",
        articles=[Article(number="1", title="Тест", anchor="z1", text="текст статьи")],
    )


def _conn(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.create_schema(conn)
    return conn


def test_prune_removed_sources_deletes_act_whose_url_left_sources_yaml(tmp_path):
    """Регрессия (Finding 2): исправление URL в sources.yaml (например, замена
    ошибочно указанного Налогового кодекса 2017 года на актуальный) не
    должно оставлять старый акт в базе навсегда рядом с новым."""
    conn = _conn(tmp_path)
    indexer.index_act(
        conn, _act("tax_code_old", "https://adilet.zan.kz/rus/docs/K1700000120"),
        embed_fn=_fake_embed,
    )
    indexer.index_act(
        conn, _act("civil_code_general", "https://adilet.zan.kz/rus/docs/K940001000_"),
        embed_fn=_fake_embed,
    )

    current_sources = [
        Source(key="civil_code_general", title="Гражданский кодекс",
               url="https://adilet.zan.kz/rus/docs/K940001000_", doc_type="кодекс"),
    ]
    removed = cli._prune_removed_sources(conn, current_sources)

    assert removed == ["Акт «tax_code_old»"]
    assert conn.execute("SELECT COUNT(*) FROM acts").fetchone()[0] == 1
    assert conn.execute(
        "SELECT adilet_url FROM acts"
    ).fetchone()[0] == "https://adilet.zan.kz/rus/docs/K940001000_"


def test_prune_removed_sources_cascades_chunks_and_vectors(tmp_path):
    """Каскад должен вычистить chunks/vec_chunks вслед за удалённым актом —
    иначе в базе остаются осиротевшие строки."""
    conn = _conn(tmp_path)
    indexer.index_act(
        conn, _act("stale", "https://adilet.zan.kz/rus/docs/STALE"), embed_fn=_fake_embed,
    )

    cli._prune_removed_sources(conn, [])

    assert conn.execute("SELECT COUNT(*) FROM acts").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0] == 0


def test_prune_removed_sources_keeps_acts_still_in_sources(tmp_path):
    """Не должен трогать акты, чей URL по-прежнему в sources.yaml."""
    conn = _conn(tmp_path)
    indexer.index_act(
        conn, _act("civil_code_general", "https://adilet.zan.kz/rus/docs/K940001000_"),
        embed_fn=_fake_embed,
    )
    current_sources = [
        Source(key="civil_code_general", title="Гражданский кодекс",
               url="https://adilet.zan.kz/rus/docs/K940001000_", doc_type="кодекс"),
    ]

    removed = cli._prune_removed_sources(conn, current_sources)

    assert removed == []
    assert conn.execute("SELECT COUNT(*) FROM acts").fetchone()[0] == 1


def test_status_reports_sources_missing_from_index(tmp_path, monkeypatch, capsys):
    """Finding 2: индекс, собранный лишь частично (скрейп/индексация упали на
    части источников и продолжили корпус — см. cmd_scrape/index_directory),
    не должен выглядеть как полный. cmd_status обязан сверить акты в индексе
    со списком sources.yaml и назвать недостающие."""
    conn = _conn(tmp_path)
    indexer.index_act(
        conn, _act("civil_code_general", "https://adilet.zan.kz/rus/docs/K940001000_"),
        embed_fn=_fake_embed,
    )
    monkeypatch.setattr(cli.db, "connect", lambda: conn)
    monkeypatch.setattr(
        cli,
        "load_sources",
        lambda: [
            Source(key="civil_code_general", title="Акт «civil_code_general»",
                   url="https://adilet.zan.kz/rus/docs/K940001000_", doc_type="кодекс"),
            Source(key="labour_code", title="Трудовой кодекс Республики Казахстан",
                   url="https://adilet.zan.kz/rus/docs/K1500000414", doc_type="кодекс"),
        ],
    )

    exit_code = cli.cmd_status()

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "Трудовой кодекс Республики Казахстан" in output
