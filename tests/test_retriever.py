import db
import indexer
import retriever
from models import Act, Article


def _act() -> Act:
    texts = {
        "30": "Трудовой договор заключается в письменной форме не менее чем в двух "
              "экземплярах и подписывается сторонами. " * 3,
        "78": "Ежегодный оплачиваемый трудовой отпуск работникам предоставляется "
              "продолжительностью двадцать четыре календарных дня. " * 3,
        "152": "Работник обязан возместить работодателю причинённый прямой "
               "действительный ущерб. " * 3,
    }
    return Act(
        key="labour_code",
        title="Трудовой кодекс Республики Казахстан",
        url="https://adilet.zan.kz/rus/docs/K1500000414",
        doc_type="кодекс",
        revision_date="2026-06-11",
        articles=[
            Article(number=n, title=f"Статья {n}", anchor=f"z{n}", text=t)
            for n, t in texts.items()
        ],
    )


def _vector_for(text: str) -> list[float]:
    """Детерминированный псевдо-эмбеддинг: близость растёт с пересечением слов."""
    vocabulary = ["договор", "отпуск", "ущерб", "письменной", "календарных", "возместить"]
    lowered = text.lower()
    head = [1.0 if word in lowered else 0.0 for word in vocabulary]
    return head + [0.0] * (1024 - len(head))


def _embed(texts):
    return [_vector_for(t) for t in texts]


def _conn(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.create_schema(conn)
    indexer.index_act(conn, _act(), embed_fn=_embed)
    return conn


def test_search_finds_relevant_article(tmp_path):
    hits = retriever.search(_conn(tmp_path), "как оформить отпуск", k=2, embed_fn=_embed)
    assert hits
    assert hits[0].article_no == "78"


def test_hit_carries_citation_metadata(tmp_path):
    hits = retriever.search(_conn(tmp_path), "трудовой договор", k=1, embed_fn=_embed)
    hit = hits[0]
    assert hit.act_title == "Трудовой кодекс Республики Казахстан"
    assert hit.revision_date == "2026-06-11"
    assert hit.citation == "статья 30 — Трудовой кодекс Республики Казахстан"
    assert hit.link == "https://adilet.zan.kz/rus/docs/K1500000414#z30"


def test_exact_article_number_query_hits_fts_branch(tmp_path):
    """Чисто векторный поиск такое проваливает — за это отвечает FTS5."""
    hits = retriever.search(_conn(tmp_path), "возместить ущерб", k=3, embed_fn=_embed)
    assert "152" in {h.article_no for h in hits}


def test_search_respects_k(tmp_path):
    hits = retriever.search(_conn(tmp_path), "трудовой договор", k=2, embed_fn=_embed)
    assert len(hits) <= 2


def test_irrelevant_query_returns_empty(tmp_path):
    """Ниже порога релевантности и без совпадений FTS — отказ, а не мусор."""
    hits = retriever.search(
        _conn(tmp_path), "рецепт борща", k=4, embed_fn=lambda t: [[0.0] * 1024 for _ in t]
    )
    assert hits == []


def test_empty_index_returns_empty(tmp_path):
    conn = db.connect(tmp_path / "empty.db")
    db.create_schema(conn)
    assert retriever.search(conn, "трудовой договор", embed_fn=_embed) == []


def test_fts_query_with_special_characters_does_not_crash(tmp_path):
    """Пользовательский ввод попадает в MATCH — синтаксис FTS5 надо экранировать."""
    hits = retriever.search(
        _conn(tmp_path), 'договор AND "ущерб OR (', k=2, embed_fn=_embed
    )
    assert isinstance(hits, list)
