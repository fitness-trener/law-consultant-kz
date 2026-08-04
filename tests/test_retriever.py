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


def test_stopword_only_match_returns_empty(tmp_path):
    """FTS-совпадение только по стоп-слову не должно спасать запрос от отказа.

    "чем" входит в текст статьи 30 ("не менее чем в двух экземплярах") и до
    фикса проходит фильтр длины (len > 2), поэтому FTS находит статью 30 —
    хотя ни один словарный признак псевдо-эмбеддинга не совпадает с запросом,
    и векторный порог не пройден.
    """
    hits = retriever.search(_conn(tmp_path), "чем", k=4, embed_fn=_embed)
    assert hits == []


def test_stopwords_only_query_returns_empty_without_crashing(tmp_path):
    """Запрос из одних стоп-слов не должен падать и не должен ничего находить."""
    hits = retriever.search(_conn(tmp_path), "что как это", k=4, embed_fn=_embed)
    assert hits == []


def _dedupe_act() -> Act:
    """Статья 123 растянута так, что режется минимум на два чанка — они не
    должны вытеснять из выдачи остальные, различные по смыслу статьи."""
    ущерб_sentence = (
        "Работник несёт материальную ответственность за причинённый "
        "работодателю прямой действительный ущерб. "
    )
    texts = {
        "123": ущерб_sentence * 30,
        "10": "Трудовой договор заключается в письменной форме не менее чем в двух "
              "экземплярах и подписывается сторонами. " * 3,
        "11": "Ежегодный оплачиваемый трудовой отпуск работникам предоставляется "
              "продолжительностью двадцать четыре календарных дня. " * 3,
        "12": "Работник обязан возместить работодателю понесённые расходы. " * 3,
    }
    return Act(
        key="labour_code_dedupe",
        title="Трудовой кодекс Республики Казахстан",
        url="https://adilet.zan.kz/rus/docs/K1500000414",
        doc_type="кодекс",
        revision_date="2026-06-11",
        articles=[
            Article(number=n, title=f"Статья {n}", anchor=f"z{n}", text=t)
            for n, t in texts.items()
        ],
    )


def test_dedup_keeps_one_chunk_per_article(tmp_path):
    """Несколько чанков одной статьи не должны занимать несколько мест в выдаче."""
    conn = db.connect(tmp_path / "dedup.db")
    db.create_schema(conn)
    indexer.index_act(conn, _dedupe_act(), embed_fn=_embed)

    hits = retriever.search(conn, "ущерб", k=4, embed_fn=_embed)

    article_numbers = [h.article_no for h in hits]
    assert len(article_numbers) == len(set(article_numbers))
    assert len(hits) == 4


def test_ubiquitous_domain_word_does_not_bypass_relevance_floor(tmp_path):
    """FTS-совпадение по слову, которое встречается почти в каждом чанке
    корпуса ("статья" — часть заголовка любой статьи), не должно спасать
    запрос от отказа. Оно не стоп-слово языка, но стало стоп-словом именно
    для этого корпуса — прежний код спасал любой запрос с таким словом."""
    hits = retriever.search(
        _conn(tmp_path),
        "рецепт борща статья",
        k=4,
        embed_fn=lambda texts: [[0.0] * 1024 for _ in texts],
    )
    assert hits == []


def test_rare_term_still_rescued_by_fts_when_vector_is_weak(tmp_path):
    """В отличие от вездесущего слова, термин, редкий для корпуса (доля
    чанков с ним ниже DISCRIMINATIVE_DF_FRACTION), — реальный сигнал
    релевантности и должен пробивать порог даже при слабом векторе."""
    filler = [
        Article(
            number=str(i),
            title=f"Статья {i}",
            anchor=f"z{i}",
            text=f"Общая норма трудового законодательства номер {i}. " * 3,
        )
        for i in range(1, 12)
    ]
    target = Article(
        number="152",
        title="Статья 152",
        anchor="z152",
        text="Работник обязан возместить работодателю причинённый прямой "
             "действительный ущерб. " * 3,
    )
    act = Act(
        key="labour_code_rare_term",
        title="Трудовой кодекс Республики Казахстан",
        url="https://adilet.zan.kz/rus/docs/K1500000414",
        doc_type="кодекс",
        revision_date="2026-06-11",
        articles=filler + [target],
    )

    # Каждый чанк получает свою координату при индексации (счётчик идёт по
    # порядку статей), а запрос — заведомо другую: косинусная близость к
    # любому чанку равна 0, спасти запрос может только FTS.
    counter = iter(range(12))

    def index_embed(texts):
        vectors = []
        for _ in texts:
            slot = next(counter)
            vectors.append([1.0 if i == slot else 0.0 for i in range(1024)])
        return vectors

    def query_embed(texts):
        return [[1.0 if i == 500 else 0.0 for i in range(1024)] for _ in texts]

    conn = db.connect(tmp_path / "rare.db")
    db.create_schema(conn)
    indexer.index_act(conn, act, embed_fn=index_embed)

    hits = retriever.search(conn, "возместить", k=4, embed_fn=query_embed)
    assert "152" in {h.article_no for h in hits}


def test_blank_query_returns_empty_without_touching_model(tmp_path):
    """Пустой или из одних пробелов запрос — вырожденный ввод, а не низкая
    релевантность. Отказ должен быть мгновенным, без обращения к модели."""

    def _boom(texts):
        raise AssertionError("embed_fn не должен вызываться для пустого запроса")

    conn = _conn(tmp_path)
    assert retriever.search(conn, "", embed_fn=_boom) == []
    assert retriever.search(conn, "   ", embed_fn=_boom) == []


def _many_chunk_article() -> Act:
    """Статья 16 ТК РК даёт 37 чанков на реальном корпусе — здесь имитируем
    статью с числом чанков, заведомо превышающим прежний CANDIDATES=20."""
    points = "\n".join(
        f"{i}. Работник несёт материальную ответственность за причинённый "
        f"работодателю прямой действительный ущерб номер {i}. "
        for i in range(1, 26)
    )
    texts = {
        "16": points,
        "10": "Трудовой договор заключается в письменной форме не менее чем в двух "
              "экземплярах и подписывается сторонами ущерб. " * 3,
        "11": "Ежегодный оплачиваемый трудовой отпуск работникам предоставляется "
              "продолжительностью двадцать четыре календарных дня ущерб. " * 3,
        "12": "Работник обязан возместить работодателю понесённые расходы ущерб. " * 3,
    }
    return Act(
        key="labour_code_many_chunks",
        title="Трудовой кодекс Республики Казахстан",
        url="https://adilet.zan.kz/rus/docs/K1500000414",
        doc_type="кодекс",
        revision_date="2026-06-11",
        articles=[
            Article(number=n, title=f"Статья {n}", anchor=f"z{n}", text=t)
            for n, t in texts.items()
        ],
    )


def test_large_article_does_not_starve_dedup(tmp_path):
    """Один длинный акт с 25+ чанками не должен вытеснять из пула кандидатов
    все остальные статьи ещё до дедупликации — иначе k=4 отдаёт 1 статью."""
    conn = db.connect(tmp_path / "many.db")
    db.create_schema(conn)
    indexer.index_act(conn, _many_chunk_article(), embed_fn=_embed)

    hits = retriever.search(conn, "ущерб", k=4, embed_fn=_embed)

    article_numbers = {h.article_no for h in hits}
    assert len(article_numbers) == 4
