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


def test_tiny_article_is_kept():
    """Отсылочные нормы короткие, но выбрасывать их нельзя."""
    chunks = chunk_article(_article("Исключена."))
    assert len(chunks) == 1
    assert "Исключена" in chunks[0].text


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
