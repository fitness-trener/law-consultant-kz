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
# adilet использует минимум две вёрстки для заголовка статьи: новые акты
# (Трудовой кодекс) — <p><b>Статья N. …</b></p>; старые (Гражданский кодекс) —
# <h3>Статья N. …</h3>. При этом <h3> у части актов несёт заголовки глав и
# разделов, а часть актов помечает их так же, как заголовок статьи — жирным
# текстом в <p> (см. «Глава 18» Трудового кодекса). Поэтому заголовок
# классифицируется не по тегу, а по тексту: совпадение с ARTICLE_RE — это
# статья, совпадение с CHAPTER_HEADING_RE — структурный заголовок (не должен
# попадать в текст статьи и не должен открывать новую статью), всё
# остальное — тело текущей статьи.
CHAPTER_HEADING_RE = re.compile(
    r"^(Глава|Раздел|РАЗДЕЛ|Часть|ЧАСТЬ|Параграф|Подраздел|"
    r"ОБЩАЯ\s+ЧАСТЬ|ОСОБЕННАЯ\s+ЧАСТЬ)(\s|$)"
)
ACT_META_RE = re.compile(
    r"от\s+(\d{1,2})\s+(\w+)\s+(\d{4})\s+года\s+№\s*([\d\-IVX]+)"
)
AMENDMENT_DATE_RE = re.compile(r"от\s+(\d{2})\.(\d{2})\.(\d{4})")
# Маркер утраты силы актом целиком. adilet не помечает это в <h1> или
# div.container_alpha.slogan — там утративший силу акт неотличим от
# действующего (проверено на живой странице K1700000120, см.
# scripts/make_fixture.py:make_repealed_tax_code_fixture). Настоящий маркер —
# сноска «Утратил силу …» в самом начале div.text_upd, ДО заголовка статьи 1
# (см. _extract_preamble). Проверять по всему тексту документа нельзя: там
# полно постатейных сносок «признаётся утратившим силу» о конкретных
# статьях — это нормально для действующих актов.
REPEALED_RE = re.compile(r"утрати[а-я]*\s+сил[а-я]*", re.IGNORECASE)
REQUEST_DELAY_SECONDS = 2.0

MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11,
    "декабря": 12,
}


class ScrapeError(RuntimeError):
    """Страница не соответствует ожидаемой структуре."""


def _heading_text(tag: Tag) -> str | None:
    """Текст, который может оказаться заголовком статьи/главы/раздела/части —
    для <p> это только текст внутри <b> (обычный текст в <p> заголовком не
    считается, иначе неразмеченное «Статья N.» в теле статьи ложно сошло бы
    за заголовок — см. test_parse_act_raises_when_article_heading_is_swallowed);
    для <h3> — весь текст элемента, вне зависимости от разметки внутри."""
    if tag.name == "h3":
        return tag.get_text(" ", strip=True)
    if tag.name == "p":
        bold = tag.find("b")
        if bold is not None:
            return bold.get_text(" ", strip=True)
    return None


def _heading_match(tag: Tag) -> re.Match[str] | None:
    """Проверяет, открывает ли тег новую статью — по тексту «Статья N.»,
    независимо от того, оформлен он как <p><b>…</b></p> или как <h3>.

    Якорь <a name="…"> / id внутри заголовка не требуется: если статье
    предшествует заметка «Примечание ИЗПИ!», adilet вешает якорь на абзац
    заметки, а не на сам заголовок статьи. Третья форма (см. _own_anchor и
    parse_act) — id самого заголовочного <p>, без <a name> внутри вовсе
    (Закон о банках, статьи 52-2/52-3: <p id="z113"><b>Статья 52-2. …</b></p>).
    """
    text = _heading_text(tag)
    if text is None:
        return None
    return ARTICLE_RE.match(text)


def _is_chapter_heading(tag: Tag) -> bool:
    """Заголовок главы/раздела/части — «Глава N.», «Раздел N.» и т.п. —
    оформленный так же, как заголовок статьи (см. _heading_text). Такой
    заголовок не должен попадать в тело статьи и не должен открывать новую
    статью."""
    text = _heading_text(tag)
    if text is None:
        return False
    return bool(CHAPTER_HEADING_RE.match(text))


def _own_anchor(tag: Tag) -> str | None:
    """Якорь тега вне контекста заголовка статьи: <a name="…"> в составе
    (форма <p><b>…</b></p>) либо id самого элемента для <h3> (форма
    <h3 id="…">, см. Гражданский кодекс). Для <p> id элемента здесь
    намеренно не читается: обычные абзацы текста статьи тоже несут id
    (см. tests/fixtures/civil_code_h3_articles_sample.html, <p id="z61">
    и т.п.) — если брать его отсюда, last_anchor начал бы наследовать id
    случайных абзацев тела статьи, а не только якоря заметок ИЗПИ вне
    заголовка. Форма C (id на самом заголовочном <p>) разбирается отдельно
    в parse_act, только когда тег уже опознан как заголовок статьи."""
    anchor = tag.find("a", attrs={"name": True})
    if anchor is not None:
        return anchor["name"]
    if tag.name == "h3":
        return tag.get("id")
    return None


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
    # Текст до заголовка статьи 1 — сюда попадает сноска «Утратил силу…»,
    # если акт утратил силу целиком (см. REPEALED_RE ниже / Finding 3).
    preamble: list[str] = []

    # <h3> у одних актов несёт заголовок статьи, у других — заголовок главы/
    # раздела/части (см. _heading_text/_is_chapter_heading — классификация
    # идёт по тексту, а не по тегу). <font> — только якорь заметки
    # «Примечание ИЗПИ!», к тексту статьи отношения не имеет. В буфер статьи
    # идут только строки из <p>, и то не из тех <p>, что сами оказались
    # заголовком статьи или главы/раздела/части.
    for tag in content.find_all(["p", "h3", "font"]):
        own_anchor = _own_anchor(tag)
        heading = _heading_match(tag)

        if heading:
            # Форма C: заголовок — <p id="…"><b>Статья N. …</b></p>, якоря
            # <a name="…"> внутри жирного текста нет вовсе, а id стоит прямо
            # на самом заголовочном <p> (см. _own_anchor: тот же id на
            # обычном абзаце тела статьи туда намеренно не попадает).
            heading_own_id = tag.get("id") if tag.name == "p" else None
            anchor = own_anchor or heading_own_id or last_anchor or ""
            # Якорь потреблён: он относится только к статье, которая его
            # только что забрала (свой либо унаследованный от заметки ИЗПИ).
            # Не сбросить здесь — значит следующая статья без своего якоря
            # молча унаследует чужой (Finding 1: 92 статьи КоАП делили якорь
            # с соседями, ссылки вели не туда).
            last_anchor = None
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
        elif current is None:
            preamble.append(tag.get_text(" ", strip=True))
        elif tag.name == "p" and not _is_chapter_heading(tag):
            line = tag.get_text(" ", strip=True)
            if line:
                buffer.append(line)

        # Не взводить last_anchor из якоря самой статьи — он уже потреблён
        # этой статьёй выше и не должен остаться «на подхвате» для
        # следующей. last_anchor нужен только для якорей, повисших на
        # заметках вне заголовка (см. docstring _heading_match).
        if own_anchor and not heading:
            last_anchor = own_anchor

    if current is not None:
        current.text = "\n".join(buffer).strip()
        articles.append(current)

    # Акт, утративший силу целиком, не должен попасть в корпус как
    # действующий (Finding 3). adilet не помечает это никак особо в <h1> или
    # в div.container_alpha.slogan — оба неотличимы от действующего акта
    # (проверено на живой странице K1700000120 — Налоговый кодекс 2017 года).
    # Настоящий маркер — сноска «Утратил силу …» перед заголовком статьи 1
    # (preamble, собранный выше). Проверять по всему тексту документа нельзя:
    # он насыщен постатейными сносками «признаётся утратившим силу» о
    # конкретных статьях — это нормально для действующих актов.
    slogan = soup.select_one("div.container_alpha.slogan")
    repeal_scope = " ".join([
        h1.get_text(" ", strip=True),
        slogan.get_text(" ", strip=True) if slogan else "",
        " ".join(preamble),
    ])
    if REPEALED_RE.search(repeal_scope):
        raise ScrapeError(
            f"{source.url}: акт признан утратившим силу и не действует"
        )

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
    # № 414-V») лежат рядом с <h1>, в div.container_alpha.slogan (уже найден
    # выше, для проверки утраты силы) — вне div.text_upd. Ищем сначала там
    # (единственное место, где это точно реквизиты самого акта, а не
    # случайное совпадение регэкспа где-то ещё на странице), и только если
    # блока нет — по всей странице.
    meta_text = slogan.get_text(" ", strip=True) if slogan else soup.get_text(" ", strip=True)
    act_date, act_number = _parse_act_meta(meta_text)

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
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
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
            try:
                page.wait_for_selector(CONTENT_SELECTOR, timeout=60_000)
            except PlaywrightTimeoutError as exc:
                raise ScrapeError(
                    f"{url}: страница загрузилась (код 200), но ожидаемая "
                    f"структура документа ({CONTENT_SELECTOR}) не появилась. "
                    "Возможно, акт признан утратившим силу, произошёл "
                    "редирект на другую страницу или изменилась вёрстка adilet"
                ) from exc
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
