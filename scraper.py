"""Загрузка и разбор страниц ИПС «Әділет» (adilet.zan.kz).

Страницы отдаются сервером целиком, но загрузка идёт через Playwright:
браузер переживает JS-проверки и антибот-заглушки, которые обычный
HTTP-клиент не проходит.
"""
from __future__ import annotations

import dataclasses
import json
import re
import time
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from config import Source
from models import Act, Article

CONTENT_SELECTOR = "div.text_upd"
ARTICLE_RE = re.compile(r"^Статья\s+(\d+(?:-\d+)?)\.\s*(.*)$", re.S)
ACT_META_RE = re.compile(
    r"от\s+(\d{1,2})\s+(\w+)\s+(\d{4})\s+года\s+№\s*([\d\-IVX]+)"
)
AMENDMENT_DATE_RE = re.compile(r"от\s+(\d{2})\.(\d{2})\.(\d{4})")
REQUEST_DELAY_SECONDS = 2.0

MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11,
    "декабря": 12,
}


class ScrapeError(RuntimeError):
    """Страница не соответствует ожидаемой структуре."""


def _heading_match(tag: Tag) -> re.Match[str] | None:
    """Проверяет, открывает ли <p> новую статью — по жирному тексту «Статья N.».

    Якорь <a name="…"> внутри жирного текста не требуется: если статье
    предшествует заметка «Примечание ИЗПИ!», adilet вешает якорь на абзац
    заметки, а не на сам заголовок статьи.
    """
    if tag.name != "p":
        return None
    bold = tag.find("b")
    if bold is None:
        return None
    return ARTICLE_RE.match(bold.get_text(" ", strip=True))


def _own_anchor(tag: Tag) -> str | None:
    """Якорь <a name="…"> из состава самого тега, если есть."""
    anchor = tag.find("a", attrs={"name": True})
    return anchor["name"] if anchor is not None else None


def _parse_act_meta(text: str) -> tuple[str | None, str | None]:
    match = ACT_META_RE.search(text)
    if not match:
        return None, None
    day, month_name, year, number = match.groups()
    month = MONTHS.get(month_name.lower())
    if month is None:
        return None, number
    return date(int(year), month, int(day)).isoformat(), number


def _parse_revision_date(text: str) -> str | None:
    """Максимальная дата изменения из сносок."""
    dates = [
        date(int(y), int(m), int(d)).isoformat()
        for d, m, y in AMENDMENT_DATE_RE.findall(text)
        if 1 <= int(m) <= 12 and 1 <= int(d) <= 31
    ]
    return max(dates) if dates else None


def parse_act(html: str, source: Source) -> Act:
    soup = BeautifulSoup(html, "lxml")

    h1 = soup.find("h1")
    if h1 is None or not h1.get_text(strip=True):
        raise ScrapeError(f"{source.url}: не найден заголовок документа (<h1>)")

    content = soup.select_one(CONTENT_SELECTOR) or soup.body
    if content is None:
        raise ScrapeError(f"{source.url}: не найдено тело документа")

    articles: list[Article] = []
    current: Article | None = None
    buffer: list[str] = []
    # Самый последний увиденный якорь <a name="…"> — используется, когда
    # заголовок статьи сам якоря не несёт (см. _heading_match).
    last_anchor: str | None = None

    # <font> нужен только чтобы поймать якорь заметки «Примечание ИЗПИ!»,
    # которая лежит вне <p> — её текст в буфер статьи не идёт (как и раньше).
    for tag in content.find_all(["p", "h3", "font"]):
        own_anchor = _own_anchor(tag)
        heading = _heading_match(tag)

        if heading:
            anchor = own_anchor or last_anchor or ""
            if current is not None:
                current.text = "\n".join(buffer).strip()
                articles.append(current)
            current = Article(
                number=heading.group(1),
                title=heading.group(2).strip(),
                anchor=anchor,
                text="",
            )
            buffer = []
        elif current is not None and tag.name != "font":
            line = tag.get_text(" ", strip=True)
            if line:
                buffer.append(line)

        if own_anchor:
            last_anchor = own_anchor

    if current is not None:
        current.text = "\n".join(buffer).strip()
        articles.append(current)

    if not articles:
        raise ScrapeError(
            f"{source.url}: не найдено ни одной статьи. "
            "Вероятно, изменилась вёрстка adilet — проверьте селекторы."
        )

    for art in articles:
        for line in art.text.split("\n"):
            if ARTICLE_RE.match(line):
                raise ScrapeError(
                    f"{source.url}: в тексте статьи {art.number} обнаружен "
                    "заголовок другой статьи — парсер пропустил границу между "
                    "статьями (проверьте вёрстку adilet)"
                )

    full_text = content.get_text(" ", strip=True)
    # Дата и номер акта («Кодекс Республики Казахстан от 23 ноября 2015 года
    # № 414-V») лежат рядом с <h1>, в div.container_alpha.slogan — вне
    # div.text_upd, поэтому ищем их по всей странице, а не только в теле.
    act_date, act_number = _parse_act_meta(soup.get_text(" ", strip=True))

    return Act(
        key=source.key,
        title=h1.get_text(" ", strip=True),
        url=source.url,
        doc_type=source.doc_type,
        act_date=act_date,
        act_number=act_number,
        revision_date=_parse_revision_date(full_text),
        articles=articles,
    )


def fetch_html(url: str) -> str:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
                )
            )
            response = page.goto(url, wait_until="domcontentloaded", timeout=120_000)
            if response is None or response.status != 200:
                status = "нет ответа" if response is None else response.status
                raise ScrapeError(f"{url}: сервер вернул {status}")
            page.wait_for_selector(CONTENT_SELECTOR, timeout=60_000)
            return page.content()
        finally:
            browser.close()


def act_to_json(act: Act) -> str:
    return json.dumps(dataclasses.asdict(act), ensure_ascii=False, indent=1)


def act_from_json(data: str) -> Act:
    raw = json.loads(data)
    articles = [Article(**a) for a in raw.pop("articles")]
    return Act(**raw, articles=articles)


def scrape_source(source: Source, out_dir: Path) -> Path:
    html = fetch_html(source.url)
    act = parse_act(html, source)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source.key}.json"
    out_path.write_text(act_to_json(act), encoding="utf-8")
    time.sleep(REQUEST_DELAY_SECONDS)  # не долбим adilet без пауз
    return out_path
