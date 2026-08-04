import re
from pathlib import Path

import pytest

from config import Source
from scraper import ScrapeError, act_from_json, act_to_json, parse_act

FIXTURE = Path(__file__).parent / "fixtures" / "labour_code_sample.html"
IZPI_FIXTURE = Path(__file__).parent / "fixtures" / "labour_code_izpi_sample.html"
CHAPTER_FIXTURE = Path(__file__).parent / "fixtures" / "labour_code_chapter_heading_sample.html"

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


@pytest.fixture
def chapter_heading_html() -> str:
    return CHAPTER_FIXTURE.read_text(encoding="utf-8")


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


def test_chapter_heading_does_not_pollute_preceding_article(chapter_heading_html):
    """Регрессия: между статьёй 14 и статьёй 15 стоит заголовок главы
    <h3>Глава 2. ГОСУДАРСТВЕННОЕ РЕГУЛИРОВАНИЕ...</h3>. Он не должен
    прилипать к тексту статьи 14, а статья 15 должна распарситься отдельно
    со своим номером."""
    act = parse_act(chapter_heading_html, SOURCE)
    article14 = next(a for a in act.articles if a.number == "14")
    assert "Глава 2" not in article14.text
    article15 = next((a for a in act.articles if a.number == "15"), None)
    assert article15 is not None, "статья 15 должна распарситься отдельной статьёй"


def test_fetch_html_raises_scrape_error_when_content_missing(monkeypatch):
    """Структурно валидная страница (200, есть <h1>) без div.text_upd —
    например, акт признан утратившим силу, либо soft-404 — должна давать
    понятную ScrapeError на русском с URL, а не сырую английскую ошибку
    Playwright из page.wait_for_selector. Держим офлайн: подменяем
    playwright.sync_api фейком вместо запуска настоящего браузера."""
    import sys

    import scraper as scraper_module

    class FakeTimeoutError(Exception):
        pass

    class FakeResponse:
        status = 200

    class FakePage:
        def goto(self, url, wait_until=None, timeout=None):
            return FakeResponse()

        def wait_for_selector(self, selector, timeout=None):
            raise FakeTimeoutError(f"Timeout {timeout}ms exceeded waiting for {selector}")

        def content(self):
            return "<html><body><h1>Тест</h1></body></html>"

    class FakeBrowser:
        def new_page(self, user_agent=None):
            return FakePage()

        def close(self):
            pass

    class FakeChromium:
        def launch(self, headless=True):
            return FakeBrowser()

    class FakePlaywrightCtx:
        chromium = FakeChromium()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    fake_module = type(
        "playwright_sync_api_stub",
        (),
        {
            "sync_playwright": staticmethod(lambda: FakePlaywrightCtx()),
            "TimeoutError": FakeTimeoutError,
        },
    )
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_module)

    url = "https://adilet.zan.kz/rus"
    with pytest.raises(ScrapeError, match=re.escape(url)):
        scraper_module.fetch_html(url)


def test_parse_act_extracts_date_and_number_from_slogan_block():
    """Дата и номер акта должны читаться из div.container_alpha.slogan, а не
    из первого попавшегося совпадения по всей странице — здесь перед
    slogan-блоком нарочно стоит совпадающий по формату текст-приманка, чтобы
    поиск «первое совпадение в документе» дал заведомо неверный результат."""
    html = (
        "<html><body>"
        "<h1>Тестовый кодекс</h1>"
        "<nav>Раздел «Похожие документы»: см. также акт от 1 марта 1999 года "
        "№ 999-I, утративший силу.</nav>"
        '<div class="container_alpha slogan">'
        "<p>Кодекс Республики Казахстан от 5 января 2020 года № 1-VI.</p>"
        "</div>"
        '<div class="text_upd">'
        '<p><b><a name="z1"></a>Статья 1. Тест</b></p><p>текст</p>'
        "</div></body></html>"
    )
    act = parse_act(html, SOURCE)
    assert act.act_date == "2020-01-05"
    assert act.act_number == "1-VI"
