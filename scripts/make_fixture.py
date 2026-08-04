"""Готовит усечённые фикстуры страницы adilet для офлайн-тестов."""
import re
import ssl
import sys
from pathlib import Path

import httpx

URL = "https://adilet.zan.kz/rus/docs/K1500000414"
CIVIL_CODE_URL = "https://adilet.zan.kz/rus/docs/K940001000_"
REPEALED_TAX_CODE_URL = "https://adilet.zan.kz/rus/docs/K1700000120"
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
OUT = FIXTURES_DIR / "labour_code_sample.html"
IZPI_OUT = FIXTURES_DIR / "labour_code_izpi_sample.html"
CHAPTER_OUT = FIXTURES_DIR / "labour_code_chapter_heading_sample.html"
CIVIL_CODE_H3_OUT = FIXTURES_DIR / "civil_code_h3_articles_sample.html"
REPEALED_TAX_CODE_OUT = FIXTURES_DIR / "tax_code_2017_repealed_sample.html"

# якорь сразу перед «Статья N.» — так выглядят нормальные заголовки статей
HEADING_RE = re.compile(r'<a name="z(\d+)"></a>Статья ')
# любой <h3>…</h3> целиком — в старых актах (Гражданский кодекс) в этой
# форме идут и заголовки статей, и заголовки глав/разделов/частей
# (см. scraper._heading_text)
H3_TAG_RE = re.compile(r"<h3[^>]*>.*?</h3>", re.S)


def _fetch(url: str = URL) -> str:
    # ponytail: httpx's bundled certifi CA set doesn't validate adilet's chain
    # (GoGetSSL intermediate) on this machine, but the OS trust store does —
    # use it instead of disabling verification.
    return httpx.get(
        url,
        timeout=90,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0"},
        verify=ssl.create_default_context(),
    ).text


def make_basic_fixture(html: str) -> int:
    # оставляем всё до пятой статьи — этого хватает для проверок парсера
    marks = [m.start() for m in HEADING_RE.finditer(html)]
    if len(marks) < 5:
        print(f"Найдено маркеров статей: {len(marks)}, ожидалось >= 5", file=sys.stderr)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html[: marks[4]] + "</div></body></html>", encoding="utf-8")
    print(f"Фикстура записана: {OUT} ({OUT.stat().st_size} байт)")
    return 0


def make_izpi_fixture(html: str) -> int:
    """Фикстура статей 3-4 — перед статьёй 4 стоит заметка «Примечание ИЗПИ!»,
    из-за которой её якорь <a name="z4"> висит вне заголовка статьи (регрессия
    на пропуск статьи, см. scraper._heading_match)."""
    matches = {m.group(1): m.start() for m in HEADING_RE.finditer(html)}
    if "3" not in matches or "5" not in matches:
        print("Не найдены заголовки статей 3 и 5 (проверьте вёрстку)", file=sys.stderr)
        return 1
    start = html.rfind("<p", 0, matches["3"])
    end = html.rfind("<p", 0, matches["5"])
    if start == -1 or end == -1 or end <= start:
        print("Не удалось вырезать фрагмент статей 3-4", file=sys.stderr)
        return 1
    body = html[start:end]
    if "Примечание ИЗПИ" not in body or "Статья 4." not in body:
        print(
            "В вырезанном фрагменте нет ожидаемой заметки ИЗПИ перед статьёй 4 — "
            "вёрстка adilet могла измениться",
            file=sys.stderr,
        )
        return 1
    IZPI_OUT.parent.mkdir(parents=True, exist_ok=True)
    IZPI_OUT.write_text(
        "<html><body><h1>Трудовой кодекс Республики Казахстан</h1>"
        f'<div class="text_upd">{body}</div></body></html>',
        encoding="utf-8",
    )
    print(f"Фикстура записана: {IZPI_OUT} ({IZPI_OUT.stat().st_size} байт)")
    return 0


def make_chapter_heading_fixture(html: str) -> int:
    """Фикстура статей 13-15 — между статьёй 14 и статьёй 15 стоит заголовок
    <h3>Глава 2. ГОСУДАРСТВЕННОЕ РЕГУЛИРОВАНИЕ...</h3>, который должен
    заканчивать статью 14, а не прилипать к её тексту (см. scraper.parse_act,
    цикл по content.find_all)."""
    matches = {m.group(1): m.start() for m in HEADING_RE.finditer(html)}
    if "13" not in matches or "16" not in matches:
        print("Не найдены заголовки статей 13 и 16 (проверьте вёрстку)", file=sys.stderr)
        return 1
    start = html.rfind("<p", 0, matches["13"])
    end = html.rfind("<p", 0, matches["16"])
    if start == -1 or end == -1 or end <= start:
        print("Не удалось вырезать фрагмент статей 13-15", file=sys.stderr)
        return 1
    body = html[start:end]
    if "Глава 2." not in body or "Статья 14." not in body or "Статья 15." not in body:
        print(
            "В вырезанном фрагменте нет ожидаемого заголовка главы между "
            "статьями 14 и 15 — вёрстка adilet могла измениться",
            file=sys.stderr,
        )
        return 1
    CHAPTER_OUT.parent.mkdir(parents=True, exist_ok=True)
    CHAPTER_OUT.write_text(
        "<html><body><h1>Трудовой кодекс Республики Казахстан</h1>"
        f'<div class="text_upd">{body}</div></body></html>',
        encoding="utf-8",
    )
    print(f"Фикстура записана: {CHAPTER_OUT} ({CHAPTER_OUT.stat().st_size} байт)")
    return 0


def make_civil_code_h3_fixture(html: str) -> int:
    """Фикстура статей 1-12 Гражданского кодекса (Общая часть) — здесь
    заголовки статей (и глав) оформлены как <h3 id="…">, а не как
    <p><b>…</b></p> у Трудового кодекса (вёрстка старых актов, см.
    scraper._heading_text). Между статьями 11 и 12 стоит заголовок «Глава 2.
    …», который не должен становиться отдельной статьёй и не должен
    прилипать к тексту статьи 11 (см. scraper._is_chapter_heading)."""
    h3s = list(H3_TAG_RE.finditer(html))

    def _h3_text(match: re.Match[str]) -> str:
        return re.sub(r"<[^>]+>", " ", match.group(0))

    article1 = next((i for i, m in enumerate(h3s) if "Статья 1." in _h3_text(m)), None)
    article13 = next((i for i, m in enumerate(h3s) if "Статья 13." in _h3_text(m)), None)
    if article1 is None or article13 is None:
        print("Не найдены заголовки статей 1 и 13 (проверьте вёрстку)", file=sys.stderr)
        return 1

    # прихватываем структурные заголовки (ОБЩАЯ ЧАСТЬ, Раздел, Глава),
    # непосредственно предшествующие статье 1, без разрывов между <h3>
    first = article1
    while first > 0 and h3s[first - 1].end() == h3s[first].start():
        first -= 1

    start = h3s[first].start()
    end = h3s[article13].start()
    if end <= start:
        print("Не удалось вырезать фрагмент статей 1-12", file=sys.stderr)
        return 1
    body = html[start:end]
    if "Статья 1." not in body or "Глава 2." not in body or "Статья 12." not in body:
        print(
            "В вырезанном фрагменте нет ожидаемых заголовков статьи 1, главы 2 "
            "и статьи 12 — вёрстка adilet могла измениться",
            file=sys.stderr,
        )
        return 1
    CIVIL_CODE_H3_OUT.parent.mkdir(parents=True, exist_ok=True)
    CIVIL_CODE_H3_OUT.write_text(
        "<html><body><h1>Гражданский кодекс Республики Казахстан (Общая часть)</h1>"
        f'<div class="text_upd">{body}</div></body></html>',
        encoding="utf-8",
    )
    print(f"Фикстура записана: {CIVIL_CODE_H3_OUT} ({CIVIL_CODE_H3_OUT.stat().st_size} байт)")
    return 0


def make_repealed_tax_code_fixture(html: str) -> int:
    """Фикстура Налогового кодекса 2017 года (K1700000120) — утратил силу с
    01.01.2026, заменён кодексом № 214-VIII. Нужна для проверки Finding 3:
    adilet не отмечает утративший силу акт ничем особым в <h1> или
    div.container_alpha.slogan (структура там идентична действующим актам —
    проверено вручную), маркер — это сноска «Утратил силу …» в самом начале
    div.text_upd, ДО заголовка статьи 1 (у действующих актов в этом месте
    только «Примечание РЦПИ!»/«Вводится в действие», см. parse_act)."""
    marks = [m.start() for m in HEADING_RE.finditer(html)]
    if not marks:
        print("Не найден заголовок статьи 1 (проверьте вёрстку)", file=sys.stderr)
        return 1
    body_start = html.find("<body")
    if body_start == -1:
        print("Не найден <body> (проверьте вёрстку)", file=sys.stderr)
        return 1
    # с запасом внутрь статьи 1 — сама статья 1 не обязана распарситься
    # целиком, проверка утраты силы срабатывает раньше
    body = html[body_start : marks[0] + 2500]
    if "Утратил силу" not in body:
        print(
            "В вырезанном фрагменте нет ожидаемой сноски «Утратил силу» — "
            "либо вёрстка adilet изменилась, либо акт больше не отмечен так",
            file=sys.stderr,
        )
        return 1
    REPEALED_TAX_CODE_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPEALED_TAX_CODE_OUT.write_text(
        body + "</p></article></div></div></div></body></html>",
        encoding="utf-8",
    )
    print(
        f"Фикстура записана: {REPEALED_TAX_CODE_OUT} "
        f"({REPEALED_TAX_CODE_OUT.stat().st_size} байт)"
    )
    return 0


def main() -> int:
    html = _fetch()
    status = make_basic_fixture(html)
    if status != 0:
        return status
    status = make_izpi_fixture(html)
    if status != 0:
        return status
    status = make_chapter_heading_fixture(html)
    if status != 0:
        return status
    civil_html = _fetch(CIVIL_CODE_URL)
    status = make_civil_code_h3_fixture(civil_html)
    if status != 0:
        return status
    repealed_html = _fetch(REPEALED_TAX_CODE_URL)
    return make_repealed_tax_code_fixture(repealed_html)


if __name__ == "__main__":
    raise SystemExit(main())
