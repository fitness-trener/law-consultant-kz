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


def _is_article_heading(tag: Tag) -> str | None:
    """Возвращает якорь, если тег открывает новую статью."""
    bold = tag.find("b")
    if bold is None:
        return None
    anchor = bold.find("a", attrs={"name": True})
    if anchor is None:
        return None
    if not ARTICLE_RE.match(bold.get_text(" ", strip=True)):
        return None
    return anchor["name"]


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

    for tag in content.find_all(["p", "h3"]):
        anchor = _is_article_heading(tag) if tag.name == "p" else None
        if anchor:
            if current is not None:
                current.text = "\n".join(buffer).strip()
                articles.append(current)
            heading = ARTICLE_RE.match(tag.get_text(" ", strip=True))
            current = Article(
                number=heading.group(1),
                title=heading.group(2).strip(),
                anchor=anchor,
                text="",
            )
            buffer = []
        elif current is not None:
            line = tag.get_text(" ", strip=True)
            if line:
                buffer.append(line)

    if current is not None:
        current.text = "\n".join(buffer).strip()
        articles.append(current)

    if not articles:
        raise ScrapeError(
            f"{source.url}: не найдено ни одной статьи. "
            "Вероятно, изменилась вёрстка adilet — проверьте селекторы."
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
