from chunker import chunk_act, chunk_article
from models import Act, Article


def _article(text: str, number: str = "10", title: str = "Оплата труда") -> Article:
    return Article(number=number, title=title, anchor=f"z{number}", text=text)


def test_short_article_is_one_chunk():
    art = _article("Заработная плата выплачивается не реже одного раза в месяц. " * 3)
    chunks = chunk_article(art)
    assert len(chunks) == 1
    assert chunks[0].part_no is None
    assert chunks[0].article_no == "10"
    assert chunks[0].anchor == "z10"
    assert chunks[0].text.startswith("Статья 10. Оплата труда")


def test_long_article_splits_by_numbered_points():
    body = "\n".join(f"{i}. " + "норма права. " * 40 for i in range(1, 6))
    chunks = chunk_article(_article(body), max_chars=800)
    assert len(chunks) > 1
    assert [c.part_no for c in chunks] == ["1", "2", "3", "4", "5"]
    # заголовок статьи повторяется в каждом фрагменте, иначе фрагмент теряет контекст
    assert all(c.text.startswith("Статья 10. Оплата труда") for c in chunks)


def test_long_article_with_gap_in_numbering_keeps_real_point_numbers():
    """Пункт 2 исключён — номера в part_no не должны идти подряд."""
    body = "\n".join(f"{i}. " + "норма права. " * 40 for i in (1, 3, 4))
    chunks = chunk_article(_article(body), max_chars=800)
    assert len(chunks) > 1
    assert [c.part_no for c in chunks] == ["1", "3", "4"]


def test_long_article_without_numbering_splits_by_paragraphs():
    body = "\n".join("абзац без нумерации. " * 30 for _ in range(4))
    chunks = chunk_article(_article(body), max_chars=700)
    assert len(chunks) > 1
    assert [c.part_no for c in chunks] == ["1", "2", "3", "4"]


def test_oversized_single_point_is_hard_split():
    body = "1. " + "очень длинный пункт без разрывов. " * 200
    chunks = chunk_article(_article(body), max_chars=800)
    assert len(chunks) > 1
    assert all(len(c.text) <= 1000 for c in chunks)
    # Finding 4: единственный пронумерованный пункт — это всё ещё нумерация,
    # а не повод для фолбэка. Фрагменты одного пункта 1 должны все нести
    # его настоящий номер, а не выдуманную последовательную нумерацию.
    assert all(c.part_no == "1" for c in chunks)


def test_tiny_article_is_kept():
    """Отсылочные нормы короткие, но выбрасывать их нельзя."""
    chunks = chunk_article(_article("Исключена."))
    assert len(chunks) == 1
    assert "Исключена" in chunks[0].text


def test_empty_article_produces_header_only_chunk():
    """Пустой текст статьи не должен приводить к падению — остаётся заголовок."""
    chunks = chunk_article(_article(""))
    assert len(chunks) == 1
    assert chunks[0].part_no is None
    assert chunks[0].text.strip() == "Статья 10. Оплата труда"


def test_article_number_with_suffix_preserved():
    chunks = chunk_article(_article("текст нормы", number="152-1"))
    assert chunks[0].article_no == "152-1"
    assert chunks[0].text.startswith("Статья 152-1.")


def test_chunk_act_processes_all_articles():
    act = Act(
        key="labour_code",
        title="Трудовой кодекс Республики Казахстан",
        url="https://adilet.zan.kz/rus/docs/K1500000414",
        doc_type="кодекс",
        articles=[_article("текст один", "1"), _article("текст два", "2")],
    )
    chunks = chunk_act(act)
    assert len(chunks) == 2
    assert [c.article_no for c in chunks] == ["1", "2"]


def test_whitespace_is_normalized():
    art = _article("текст   с\n\n\nлишними    пробелами")
    assert "  " not in chunk_article(art)[0].text


def test_lead_in_before_first_point_is_not_dropped():
    """Finding 1: преамбула до первого пронумерованного пункта ('...следующие
    условия:') раньше не попадала ни в один кусок и терялась безвозвратно.
    """
    lead = "Трудовой договор должен содержать следующие условия:"
    body = (
        lead
        + "\n1. " + "Пункт первый содержит развёрнутую норму. " * 30
        + "\n2. " + "Пункт второй тоже развёрнутый. " * 30
    )
    chunks = chunk_article(_article(body), max_chars=900)
    assert any(lead in c.text for c in chunks)


def test_hard_split_fragment_never_exceeds_max_chars():
    """Finding 2: бюджет `_hard_split` игнорировал перевод строки между
    заголовком и телом (off-by-one), а короткий хвост клеился к предыдущему
    без проверки итогового размера — оба пути могли дать чанк больше max_chars.
    """
    art = _article(
        "Норма права устанавливает порядок применения статьи в соответствующих случаях далее. " * 70,
        title="Оплата труда",
    )
    chunks = chunk_article(art, max_chars=2000, min_chars=100)
    assert len(chunks) > 1
    assert all(len(c.text) <= 2000 for c in chunks)


def test_short_first_point_merges_forward_into_next():
    """Finding 3: если короткий пункт стоит первым, приклеивать его назад
    не к чему, и он раньше уезжал в чанк ниже min_chars. Теперь он должен
    склеиться со следующим пунктом и взять его номер.
    """
    body = (
        "1. Утратил силу.\n2. "
        + "Развернутая норма права применяется в установленном порядке. " * 40
    )
    chunks = chunk_article(_article(body))
    assert "1" not in [c.part_no for c in chunks]
    assert chunks[0].part_no == "2"
    assert "Утратил силу" in chunks[0].text


def test_no_text_is_lost_across_chunks():
    """Конкатенация всех чанков должна содержать каждое слово исходной статьи —
    это главная гарантия модуля: ретривер не должен молча терять текст закона.
    """
    body = (
        "Трудовой договор должен содержать следующие условия:\n"
        "1. Место работы. "
        + "Развёрнутая норма права применяется в установленном порядке. " * 5
        + "\n2. Оплата труда. "
        + "Норма права устанавливает порядок применения статьи. " * 40
    )
    chunks = chunk_article(_article(body), max_chars=800, min_chars=100)
    corpus = " ".join(c.text for c in chunks)
    for word in " ".join(body.split()).split(" "):
        assert word in corpus


def test_hard_split_breaks_oversized_single_sentence():
    """Критический дефект: перечисление подпунктов внутри одного предложения
    (запятые, единственная точка в конце) — обычная практика казахстанских
    НПА. `_hard_split` не проверял длину отдельного предложения и раньше
    отдавал такое предложение целиком, превышая max_chars в разы.
    """
    body = "1. " + ", ".join(f"подпункт номер {i}" for i in range(1, 250)) + "."
    chunks = chunk_article(_article(body), max_chars=2000, min_chars=100)
    assert all(len(c.text) <= 2000 for c in chunks)
    corpus = " ".join(c.text for c in chunks)
    for word in " ".join(body.split()).split(" "):
        assert word in corpus


def test_hard_split_breaks_paragraph_without_terminal_punctuation():
    """Критический дефект: абзац вовсе без `.` и `;` — re.split не находит
    ни одной границы, и весь абзац раньше уходил в один чанк без проверки
    max_chars.
    """
    body = "текст без точек и точек с запятой совсем " * 400
    chunks = chunk_article(_article(body), max_chars=2000, min_chars=100)
    assert all(len(c.text) <= 2000 for c in chunks)
    corpus = " ".join(c.text for c in chunks)
    for word in " ".join(body.split()).split(" "):
        assert word in corpus


def test_hard_split_breaks_whitespace_free_token_longer_than_budget():
    """Патологический случай: единственный токен без пробелов длиннее
    бюджета целиком (например, склеенный текст без пробелов). Делить его
    негде, кроме как по сырым символам — но и тогда ни один символ не
    должен потеряться, а каждый фрагмент обязан укладываться в max_chars.
    """
    header = "Статья 10. Оплата труда"
    token = "а" * 5000
    chunks = chunk_article(_article(token), max_chars=2000, min_chars=100)
    assert all(len(c.text) <= 2000 for c in chunks)
    bodies = [c.text[len(header):].lstrip("\n") for c in chunks]
    assert "".join(bodies) == token


def test_general_invariants_across_representative_articles():
    """Общая гарантия модуля, которую он должен держать для всего остального
    пайплайна: независимо от формы статьи — каждый чанк укладывается в
    max_chars, а конкатенация чанков сохраняет каждое слово исходного текста.
    """
    max_chars = 2000
    bodies = {
        "short": "Короткая норма права применяется в установленном порядке. " * 3,
        "long_with_points": "\n".join(
            f"{i}. "
            + "развёрнутая норма права применяется в установленном порядке. " * 40
            for i in range(1, 6)
        ),
        "long_without_points": (
            "абзац без нумерации, но с развёрнутым текстом нормы права. " * 100
        ),
        "lead_in_before_first_point": (
            "Трудовой договор должен содержать следующие условия:\n"
            "1. " + "Пункт первый содержит развёрнутую норму права. " * 40
            + "\n2. " + "Пункт второй тоже развёрнутый и подробный. " * 40
        ),
        "oversized_single_sentence": (
            "1. " + ", ".join(f"подпункт номер {i}" for i in range(1, 250)) + "."
        ),
    }
    for name, body in bodies.items():
        chunks = chunk_article(_article(body), max_chars=max_chars, min_chars=100)
        assert all(len(c.text) <= max_chars for c in chunks), name
        corpus = " ".join(c.text for c in chunks)
        for word in " ".join(body.split()).split(" "):
            assert word in corpus, (name, word)
