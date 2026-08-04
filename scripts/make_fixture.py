"""Готовит усечённую фикстуру страницы adilet для офлайн-тестов."""
import re
import ssl
import sys
from pathlib import Path

import httpx

URL = "https://adilet.zan.kz/rus/docs/K1500000414"
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "labour_code_sample.html"


def main() -> int:
    # ponytail: httpx's bundled certifi CA set doesn't validate adilet's chain
    # (GoGetSSL intermediate) on this machine, but the OS trust store does —
    # use it instead of disabling verification.
    html = httpx.get(
        URL,
        timeout=90,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0"},
        verify=ssl.create_default_context(),
    ).text
    # оставляем всё до пятой статьи — этого хватает для проверок парсера
    marks = [m.start() for m in re.finditer(r'<a name="z\d+"></a>Статья ', html)]
    if len(marks) < 5:
        print(f"Найдено маркеров статей: {len(marks)}, ожидалось >= 5", file=sys.stderr)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html[: marks[4]] + "</div></body></html>", encoding="utf-8")
    print(f"Фикстура записана: {OUT} ({OUT.stat().st_size} байт)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
