"""Готовит усечённые фикстуры страницы adilet для офлайн-тестов."""
import re
import ssl
import sys
from pathlib import Path

import httpx

URL = "https://adilet.zan.kz/rus/docs/K1500000414"
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
OUT = FIXTURES_DIR / "labour_code_sample.html"
IZPI_OUT = FIXTURES_DIR / "labour_code_izpi_sample.html"

# якорь сразу перед «Статья N.» — так выглядят нормальные заголовки статей
HEADING_RE = re.compile(r'<a name="z(\d+)"></a>Статья ')


def _fetch() -> str:
    # ponytail: httpx's bundled certifi CA set doesn't validate adilet's chain
    # (GoGetSSL intermediate) on this machine, but the OS trust store does —
    # use it instead of disabling verification.
    return httpx.get(
        URL,
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


def main() -> int:
    html = _fetch()
    status = make_basic_fixture(html)
    if status != 0:
        return status
    return make_izpi_fixture(html)


if __name__ == "__main__":
    raise SystemExit(main())
