import re
from pathlib import Path

import pytest

from config import Source
from scraper import ScrapeError, act_from_json, act_to_json, parse_act

FIXTURE = Path(__file__).parent / "fixtures" / "labour_code_sample.html"
IZPI_FIXTURE = Path(__file__).parent / "fixtures" / "labour_code_izpi_sample.html"

SOURCE = Source(
    key="labour_code",
    title="Трудовой кодекс Республики Казахстан",
    url="https://adilet.zan.kz/rus/docs/K1500000414",
    doc_type="кодекс",
)


@pytest.fixture
def html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture
def izpi_html() -> str:
    return IZPI_FIXTURE.read_text(encoding="utf-8")


def test_parse_act_reads_title(html):
    act = parse_act(html, SOURCE)
    assert "Трудовой кодекс" in act.title
    assert act.url == SOURCE.url
    assert act.doc_type == "кодекс"


def test_parse_act_extracts_articles(html):
    act = parse_act(html, SOURCE)
    assert len(act.articles) >= 3
    first = act.articles[0]
    assert first.number == "1"
    assert "понятия" in first.title.lower()
    assert first.anchor == "z1"
    assert len(first.text) > 200


def test_article_text_excludes_next_article_heading(html):
    act = parse_act(html, SOURCE)
    assert "Статья 2." not in act.articles[0].text


def test_parse_act_extracts_date_and_number(html):
    act = parse_act(html, SOURCE)
    assert act.act_date == "2015-11-23"
    assert act.act_number == "414-V"


def test_parse_act_extracts_latest_amendment_date(html):
    """revision_date — максимальная дата из сносок об изменениях."""
    act = parse_act(html, SOURCE)
    assert act.revision_date is not None
    assert act.revision_date >= "2015-11-23"


def test_parse_act_rejects_page_without_articles():
    with pytest.raises(ScrapeError, match="не найдено ни одной статьи"):
        parse_act("<html><body><h1>Пусто</h1></body></html>", SOURCE)


def test_parse_act_rejects_page_without_title():
    html = '<html><body><div class="text_upd">'
    html += '<p><b><a name="z1"></a>Статья 1. Тест</b></p><p>текст</p></div></body></html>'
    with pytest.raises(ScrapeError, match="не найден заголовок"):
        parse_act(html, SOURCE)


def test_json_roundtrip(html):
    act = parse_act(html, SOURCE)
    restored = act_from_json(act_to_json(act))
    assert restored.title == act.title
    assert len(restored.articles) == len(act.articles)
    assert restored.articles[0].text == act.articles[0].text


def test_article_after_izpi_note_is_parsed_separately(izpi_html):
    """Регрессия: перед статьёй 4 стоит заметка «Примечание ИЗПИ!», и её якорь
    <a name="z4"> висит вне заголовка статьи (не внутри <p><b>). Статья 4
    должна остаться отдельной статьёй со своим номером и непустым якорем, а
    не потеряться внутри текста статьи 3."""
    act = parse_act(izpi_html, SOURCE)
    article4 = next((a for a in act.articles if a.number == "4"), None)
    assert article4 is not None, "статья 4 должна быть отдельной статьёй"
    assert article4.anchor  # якорь должен подтянуться из заметки ИЗПИ, а не быть пустым
    article3 = next(a for a in act.articles if a.number == "3")
    assert "Принципами трудового законодательства" not in article3.text


def test_parse_act_raises_when_article_heading_is_swallowed():
    """Если парсер пропускает заголовок статьи (например, он не выделен
    <b>), её текст утекает в предыдущую статью — это должно быть громкой
    ошибкой, а не тихой порчей корпуса."""
    html = (
        '<html><body><h1>Тест</h1><div class="text_upd">'
        '<p><b><a name="z1"></a>Статья 1. Первая</b></p>'
        "<p>текст первой статьи</p>"
        "<p>Статья 2. Вторая (заголовок без выделения жирным)</p>"
        "<p>текст второй статьи</p>"
        "</div></body></html>"
    )
    with pytest.raises(ScrapeError, match=re.escape(SOURCE.url)):
        parse_act(html, SOURCE)
