from pathlib import Path

import pytest

from config import Source
from scraper import ScrapeError, act_from_json, act_to_json, parse_act

FIXTURE = Path(__file__).parent / "fixtures" / "labour_code_sample.html"

SOURCE = Source(
    key="labour_code",
    title="Трудовой кодекс Республики Казахстан",
    url="https://adilet.zan.kz/rus/docs/K1500000414",
    doc_type="кодекс",
)


@pytest.fixture
def html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


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
