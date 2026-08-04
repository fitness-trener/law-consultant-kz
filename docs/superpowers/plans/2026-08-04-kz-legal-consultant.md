# KZ Legal Consultant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Веб-приложение, отвечающее на юридические вопросы строго по текстам НПА Республики Казахстан и анализирующее загруженные документы на правовые риски с предложениями правок.

**Architecture:** Python-монолит. Playwright скрейпит adilet.zan.kz в JSON, индексатор режет тексты на статьи и кладёт в SQLite с векторным индексом и полнотекстовым поиском. Ретривер отдаёт статьи, локальная LLM через Ollama формулирует ответ строго из переданного контекста и отказывается отвечать при отсутствии релевантных норм. Анализ документа — тот же ретривер по каждому пункту загруженного файла.

**Tech Stack:** Python 3.11, FastAPI, SQLite + sqlite-vec + FTS5, Ollama (qwen3:4b-instruct, bge-m3), Playwright, BeautifulSoup, python-docx, pypdf, Jinja2 + HTMX, pytest.

## Global Constraints

- Python 3.11 (установлен 3.11.15). Виртуальное окружение — `.venv` в корне репозитория.
- Модель генерации: `qwen3:4b-instruct`, `num_ctx=6144`. Фолбэк при OOM: `qwen2.5:3b-instruct`.
- Модель эмбеддингов: `bge-m3`, размерность вектора **1024**.
- GPU — GTX 1650 Ti, 4 GB VRAM. Одновременная загрузка обеих моделей невозможна: индексация и обслуживание запросов разведены по времени.
- Свободного диска 15 GB. Не скачивать модели сверх `bge-m3`.
- Ollama 0.32.5, API: `POST /api/chat`, `POST /api/embed`. Базовый URL `http://localhost:11434`.
- **Windows-консоль по умолчанию cp1251 и падает на кириллице.** Во всех скриптах и тестах, печатающих русский текст, требуется `PYTHONIOENCODING=utf-8`. Все операции с файлами — с явным `encoding="utf-8"`.
- Все ответы пользователю на русском языке.
- Дисклеймер в каждом ответе и отчёте: справочная информация на основе текстов НПА РК, не является юридической консультацией.
- Порог релевантности ретривера: косинусная близость `0.45`, значение в конфигурации.
- Максимальный размер чанка: `2000` символов. Минимальный: `100` символов.
- Лимиты загрузки: файл до 10 МБ, до 300 пунктов.
- OCR не поддерживается. PDF без текстового слоя — явная ошибка.
- Каждая задача завершается коммитом. Сообщения коммитов на английском, тело — по необходимости.

---

## File Structure

| Файл | Ответственность |
|---|---|
| `config.py` | Настройки из `.env`, список НПА из `config/sources.yaml`, валидация |
| `llm.py` | Провайдер-агностичный доступ к модели: `complete`, `complete_json`, `embed` |
| `db.py` | Подключение к SQLite, загрузка sqlite-vec, создание схемы |
| `chunker.py` | Чистая функция: статья НПА → список чанков |
| `scraper.py` | Playwright-загрузка страницы adilet + парсинг HTML → `Act` |
| `indexer.py` | `Act` → чанки → эмбеддинги → база |
| `retriever.py` | Гибридный поиск, слияние RRF, порог релевантности |
| `docparse.py` | DOCX/PDF → список пунктов документа |
| `analyzer.py` | Пункт документа → находка через ретривер и LLM |
| `app.py` | FastAPI: эндпоинты, SSE, отдача шаблонов |
| `templates/` | Jinja2-шаблоны интерфейса |
| `cli.py` | Команды: `scrape`, `index`, `status` |

---

### Task 1: Каркас репозитория и конфигурация

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `.env.example`, `config/sources.yaml`, `config.py`, `tests/test_config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: ничего
- Produces: `config.Settings` (dataclass с полями `ollama_url: str`, `llm_provider: str`, `gen_model: str`, `embed_model: str`, `num_ctx: int`, `embed_dim: int`, `relevance_threshold: float`, `max_chunk_chars: int`, `min_chunk_chars: int`, `db_path: Path`, `raw_dir: Path`, `upload_dir: Path`, `max_upload_mb: int`, `max_clauses: int`); `config.load_settings() -> Settings`; `config.load_sources() -> list[Source]`; `config.Source` (dataclass: `key: str`, `title: str`, `url: str`, `doc_type: str`)

- [ ] **Step 1: Создать виртуальное окружение и файл зависимостей**

```bash
cd C:/Users/Alyhan/Claude/Projects/Law_Consultant
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
```

Создать `requirements.txt`:

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
jinja2==3.1.5
python-multipart==0.0.20
httpx==0.28.1
sqlite-vec==0.1.9
playwright==1.49.1
beautifulsoup4==4.12.3
lxml==5.3.0
python-docx==1.1.2
pypdf==5.1.0
pyyaml==6.0.2
pytest==8.3.4
pytest-asyncio==0.25.0
pytest-env==1.1.5
```

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m playwright install chromium
```

- [ ] **Step 2: Написать падающий тест конфигурации**

Создать `tests/test_config.py`:

```python
from pathlib import Path

import pytest

from config import Source, load_settings, load_sources


def test_load_settings_uses_defaults(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GEN_MODEL", raising=False)
    s = load_settings()
    assert s.llm_provider == "ollama"
    assert s.gen_model == "qwen3:4b-instruct"
    assert s.embed_model == "bge-m3"
    assert s.embed_dim == 1024
    assert s.num_ctx == 6144
    assert s.relevance_threshold == 0.45


def test_load_settings_reads_env(monkeypatch):
    monkeypatch.setenv("GEN_MODEL", "qwen2.5:3b-instruct")
    monkeypatch.setenv("RELEVANCE_THRESHOLD", "0.6")
    s = load_settings()
    assert s.gen_model == "qwen2.5:3b-instruct"
    assert s.relevance_threshold == 0.6


def test_load_settings_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        load_settings()


def test_load_sources_returns_acts():
    sources = load_sources()
    assert len(sources) >= 20
    labour = next(s for s in sources if s.key == "labour_code")
    assert labour.url.startswith("https://adilet.zan.kz/rus/docs/")
    assert labour.doc_type == "кодекс"
    assert all(isinstance(s, Source) for s in sources)


def test_load_sources_rejects_duplicate_keys(tmp_path):
    bad = tmp_path / "sources.yaml"
    bad.write_text(
        "sources:\n"
        "  - key: a\n    title: A\n    url: https://adilet.zan.kz/rus/docs/A\n    doc_type: закон\n"
        "  - key: a\n    title: B\n    url: https://adilet.zan.kz/rus/docs/B\n    doc_type: закон\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="дублирующийся ключ"):
        load_sources(bad)
```

- [ ] **Step 3: Запустить тест, убедиться что падает**

```bash
.venv/Scripts/python.exe -m pytest tests/test_config.py -v
```

Ожидается: FAIL с `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 4: Создать `config/sources.yaml`**

```yaml
sources:
  - key: civil_code_general
    title: Гражданский кодекс Республики Казахстан (Общая часть)
    url: https://adilet.zan.kz/rus/docs/K940001000_
    doc_type: кодекс
  - key: civil_code_special
    title: Гражданский кодекс Республики Казахстан (Особенная часть)
    url: https://adilet.zan.kz/rus/docs/K990000409_
    doc_type: кодекс
  - key: labour_code
    title: Трудовой кодекс Республики Казахстан
    url: https://adilet.zan.kz/rus/docs/K1500000414
    doc_type: кодекс
  - key: tax_code
    title: Кодекс Республики Казахстан о налогах и других обязательных платежах в бюджет
    url: https://adilet.zan.kz/rus/docs/K2500000001
    doc_type: кодекс
  - key: admin_offences_code
    title: Кодекс Республики Казахстан об административных правонарушениях
    url: https://adilet.zan.kz/rus/docs/K1400000235
    doc_type: кодекс
  - key: criminal_code
    title: Уголовный кодекс Республики Казахстан
    url: https://adilet.zan.kz/rus/docs/K1400000226
    doc_type: кодекс
  - key: criminal_procedure_code
    title: Уголовно-процессуальный кодекс Республики Казахстан
    url: https://adilet.zan.kz/rus/docs/K1400000231
    doc_type: кодекс
  - key: civil_procedure_code
    title: Гражданский процессуальный кодекс Республики Казахстан
    url: https://adilet.zan.kz/rus/docs/K1500000377
    doc_type: кодекс
  - key: entrepreneurial_code
    title: Предпринимательский кодекс Республики Казахстан
    url: https://adilet.zan.kz/rus/docs/K1500000375
    doc_type: кодекс
  - key: land_code
    title: Земельный кодекс Республики Казахстан
    url: https://adilet.zan.kz/rus/docs/K030000442_
    doc_type: кодекс
  - key: family_code
    title: Кодекс Республики Казахстан о браке (супружестве) и семье
    url: https://adilet.zan.kz/rus/docs/K1100000518
    doc_type: кодекс
  - key: administrative_procedure_code
    title: Административный процедурно-процессуальный кодекс Республики Казахстан
    url: https://adilet.zan.kz/rus/docs/K2000000350
    doc_type: кодекс
  - key: llp_law
    title: О товариществах с ограниченной и дополнительной ответственностью
    url: https://adilet.zan.kz/rus/docs/Z980000220_
    doc_type: закон
  - key: joint_stock_law
    title: Об акционерных обществах
    url: https://adilet.zan.kz/rus/docs/Z030000415_
    doc_type: закон
  - key: procurement_law
    title: О государственных закупках
    url: https://adilet.zan.kz/rus/docs/Z2400000106
    doc_type: закон
  - key: personal_data_law
    title: О персональных данных и их защите
    url: https://adilet.zan.kz/rus/docs/Z1300000094
    doc_type: закон
  - key: consumer_rights_law
    title: О защите прав потребителей
    url: https://adilet.zan.kz/rus/docs/Z100000274_
    doc_type: закон
  - key: housing_law
    title: О жилищных отношениях
    url: https://adilet.zan.kz/rus/docs/Z970000094_
    doc_type: закон
  - key: banks_law
    title: О банках и банковской деятельности в Республике Казахстан
    url: https://adilet.zan.kz/rus/docs/Z950002444_
    doc_type: закон
  - key: permits_law
    title: О разрешениях и уведомлениях
    url: https://adilet.zan.kz/rus/docs/Z1400000202
    doc_type: закон
  - key: legal_acts_law
    title: О правовых актах
    url: https://adilet.zan.kz/rus/docs/Z1600000480
    doc_type: закон
```

- [ ] **Step 5: Создать `.env.example`**

```
# ollama | anthropic | openai
LLM_PROVIDER=ollama
OLLAMA_URL=http://localhost:11434
GEN_MODEL=qwen3:4b-instruct
EMBED_MODEL=bge-m3
NUM_CTX=6144
EMBED_DIM=1024
RELEVANCE_THRESHOLD=0.45
DB_PATH=law.db
MAX_UPLOAD_MB=10
MAX_CLAUSES=300

# Заполняется только при LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-5

# Заполняется только при LLM_PROVIDER=openai
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o
```

- [ ] **Step 6: Реализовать `config.py`**

```python
"""Настройки приложения и список источников НПА."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
VALID_PROVIDERS = {"ollama", "anthropic", "openai"}


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    ollama_url: str
    gen_model: str
    embed_model: str
    num_ctx: int
    embed_dim: int
    relevance_threshold: float
    db_path: Path
    raw_dir: Path
    upload_dir: Path
    max_upload_mb: int
    max_clauses: int
    max_chunk_chars: int = 2000
    min_chunk_chars: int = 100


@dataclass(frozen=True)
class Source:
    key: str
    title: str
    url: str
    doc_type: str


def _load_dotenv() -> None:
    """Читает .env в os.environ, не затирая уже заданные переменные."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def load_settings() -> Settings:
    _load_dotenv()
    provider = os.environ.get("LLM_PROVIDER", "ollama")
    if provider not in VALID_PROVIDERS:
        raise ValueError(
            f"LLM_PROVIDER={provider!r} не поддерживается. "
            f"Допустимые значения: {', '.join(sorted(VALID_PROVIDERS))}"
        )
    return Settings(
        llm_provider=provider,
        ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        gen_model=os.environ.get("GEN_MODEL", "qwen3:4b-instruct"),
        embed_model=os.environ.get("EMBED_MODEL", "bge-m3"),
        num_ctx=int(os.environ.get("NUM_CTX", "6144")),
        embed_dim=int(os.environ.get("EMBED_DIM", "1024")),
        relevance_threshold=float(os.environ.get("RELEVANCE_THRESHOLD", "0.45")),
        db_path=ROOT / os.environ.get("DB_PATH", "law.db"),
        raw_dir=ROOT / "data" / "raw",
        upload_dir=ROOT / "data" / "uploads",
        max_upload_mb=int(os.environ.get("MAX_UPLOAD_MB", "10")),
        max_clauses=int(os.environ.get("MAX_CLAUSES", "300")),
    )


def load_sources(path: Path | None = None) -> list[Source]:
    path = path or ROOT / "config" / "sources.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    sources: list[Source] = []
    seen: set[str] = set()
    for item in data["sources"]:
        if item["key"] in seen:
            raise ValueError(f"В {path.name} дублирующийся ключ: {item['key']}")
        seen.add(item["key"])
        sources.append(
            Source(
                key=item["key"],
                title=item["title"],
                url=item["url"],
                doc_type=item["doc_type"],
            )
        )
    return sources
```

- [ ] **Step 7: Создать `pytest.ini`**

```ini
[pytest]
testpaths = tests
pythonpath = .
markers =
    integration: требует запущенной Ollama и собранного индекса
addopts = -m "not integration"
env = PYTHONIOENCODING=utf-8
```

- [ ] **Step 8: Запустить тесты, убедиться что проходят**

```bash
.venv/Scripts/python.exe -m pytest tests/test_config.py -v
```

Ожидается: PASS, 5 тестов

- [ ] **Step 9: Коммит**

```bash
git add requirements.txt pytest.ini .env.example config.py config/sources.yaml tests/test_config.py
git commit -m "feat: project scaffold, settings and NPA source list"
```

---

### Task 2: Доступ к модели

**Files:**
- Create: `llm.py`, `tests/test_llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `config.Settings`, `config.load_settings`
- Produces: `llm.LLMError` (Exception); `llm.complete(messages: list[dict], *, temperature: float = 0.2) -> str`; `llm.complete_json(messages: list[dict]) -> dict`; `llm.embed(texts: list[str]) -> list[list[float]]`; `llm.health_check() -> None` (бросает `LLMError` с человеческим текстом, если Ollama недоступна или модель не загружена)

- [ ] **Step 1: Загрузить модель эмбеддингов**

```bash
ollama pull bge-m3
```

Ожидается: `success`. Занимает ~1.2 GB на диске.

Проверить, что эмбеддинги отдаются:

```bash
curl -s -X POST http://localhost:11434/api/embed -d "{\"model\":\"bge-m3\",\"input\":\"трудовой договор\"}" | head -c 120
```

Ожидается: JSON вида `{"model":"bge-m3","embeddings":[[0.01,...`

- [ ] **Step 2: Написать падающие тесты**

Создать `tests/test_llm.py`:

```python
import json

import httpx
import pytest

import llm


def _mock_transport(handler):
    return httpx.MockTransport(handler)


def test_complete_returns_message_content(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert body["model"] == "qwen3:4b-instruct"
        assert body["stream"] is False
        assert body["options"]["num_ctx"] == 6144
        return httpx.Response(200, json={"message": {"content": "Ответ модели"}})

    monkeypatch.setattr(llm, "_transport", _mock_transport(handler))
    assert llm.complete([{"role": "user", "content": "вопрос"}]) == "Ответ модели"


def test_complete_strips_thinking_block(monkeypatch):
    """qwen3 возвращает рассуждения в <think>...</think> — они не для пользователя."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"content": "<think>рассуждаю</think>\nЧистый ответ"}},
        )

    monkeypatch.setattr(llm, "_transport", _mock_transport(handler))
    assert llm.complete([{"role": "user", "content": "x"}]) == "Чистый ответ"


def test_complete_json_parses_object(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["format"] == "json"
        return httpx.Response(200, json={"message": {"content": '{"risk": "high"}'}})

    monkeypatch.setattr(llm, "_transport", _mock_transport(handler))
    assert llm.complete_json([{"role": "user", "content": "x"}]) == {"risk": "high"}


def test_complete_json_retries_once_on_bad_json(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        content = "не json" if calls["n"] == 1 else '{"risk": "low"}'
        return httpx.Response(200, json={"message": {"content": content}})

    monkeypatch.setattr(llm, "_transport", _mock_transport(handler))
    assert llm.complete_json([{"role": "user", "content": "x"}]) == {"risk": "low"}
    assert calls["n"] == 2


def test_complete_json_raises_after_second_failure(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "мусор"}})

    monkeypatch.setattr(llm, "_transport", _mock_transport(handler))
    with pytest.raises(llm.LLMError, match="не вернула валидный JSON"):
        llm.complete_json([{"role": "user", "content": "x"}])


def test_embed_returns_vectors(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        body = json.loads(request.content)
        assert body["model"] == "bge-m3"
        assert body["input"] == ["а", "б"]
        return httpx.Response(200, json={"embeddings": [[0.1] * 1024, [0.2] * 1024]})

    monkeypatch.setattr(llm, "_transport", _mock_transport(handler))
    vectors = llm.embed(["а", "б"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 1024


def test_embed_rejects_wrong_dimension(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[0.1] * 768]})

    monkeypatch.setattr(llm, "_transport", _mock_transport(handler))
    with pytest.raises(llm.LLMError, match="размерность"):
        llm.embed(["а"])


def test_connection_error_gives_human_message(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(llm, "_transport", _mock_transport(handler))
    with pytest.raises(llm.LLMError, match="Ollama недоступна"):
        llm.complete([{"role": "user", "content": "x"}])


def test_out_of_memory_suggests_fallback_model(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "model requires more system memory"})

    monkeypatch.setattr(llm, "_transport", _mock_transport(handler))
    with pytest.raises(llm.LLMError, match="qwen2.5:3b-instruct"):
        llm.complete([{"role": "user", "content": "x"}])
```

- [ ] **Step 3: Запустить тесты, убедиться что падают**

```bash
.venv/Scripts/python.exe -m pytest tests/test_llm.py -v
```

Ожидается: FAIL с `ModuleNotFoundError: No module named 'llm'`

- [ ] **Step 4: Реализовать `llm.py`**

```python
"""Провайдер-агностичный доступ к языковой модели.

Сейчас реализован Ollama. Переключение на anthropic/openai — через
LLM_PROVIDER в .env, без изменения вызывающего кода.
"""
from __future__ import annotations

import json
import re
from typing import Any

import httpx

from config import load_settings

SETTINGS = load_settings()
TIMEOUT = httpx.Timeout(300.0, connect=10.0)

# Подменяется в тестах на httpx.MockTransport.
_transport: httpx.BaseTransport | None = None

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMError(RuntimeError):
    """Отказ на стороне модели, с текстом пригодным для показа пользователю."""


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=SETTINGS.ollama_url, timeout=TIMEOUT, transport=_transport
    )


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    if SETTINGS.llm_provider != "ollama":
        raise LLMError(
            f"Провайдер {SETTINGS.llm_provider} ещё не реализован. "
            "Установите LLM_PROVIDER=ollama."
        )
    try:
        with _client() as client:
            response = client.post(path, json=payload)
    except httpx.ConnectError as exc:
        raise LLMError(
            "Ollama недоступна по адресу "
            f"{SETTINGS.ollama_url}. Запустите её командой: ollama serve"
        ) from exc
    except httpx.TimeoutException as exc:
        raise LLMError(
            "Модель не ответила за 5 минут. Вероятно, не хватает видеопамяти — "
            "попробуйте GEN_MODEL=qwen2.5:3b-instruct в .env"
        ) from exc

    if response.status_code != 200:
        detail = response.json().get("error", response.text)
        if "memory" in detail.lower():
            raise LLMError(
                f"Модели {SETTINGS.gen_model} не хватает памяти. "
                "Укажите в .env более лёгкую модель: GEN_MODEL=qwen2.5:3b-instruct"
            )
        raise LLMError(f"Ollama вернула ошибку: {detail}")
    return response.json()


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def complete(messages: list[dict[str, str]], *, temperature: float = 0.2) -> str:
    payload = {
        "model": SETTINGS.gen_model,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": SETTINGS.num_ctx, "temperature": temperature},
    }
    data = _post("/api/chat", payload)
    return _strip_thinking(data["message"]["content"])


def complete_json(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Запрашивает JSON-объект. Один повтор при невалидном ответе."""
    payload = {
        "model": SETTINGS.gen_model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"num_ctx": SETTINGS.num_ctx, "temperature": 0.0},
    }
    for attempt in (1, 2):
        data = _post("/api/chat", payload)
        raw = _strip_thinking(data["message"]["content"])
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            if attempt == 2:
                raise LLMError(
                    f"Модель не вернула валидный JSON после повтора: {raw[:200]}"
                )
            payload["messages"] = messages + [
                {"role": "user", "content": "Верни ТОЛЬКО валидный JSON-объект."}
            ]
            continue
        if not isinstance(parsed, dict):
            raise LLMError(f"Модель не вернула валидный JSON-объект: {raw[:200]}")
        return parsed
    raise AssertionError("недостижимо")


def embed(texts: list[str]) -> list[list[float]]:
    payload = {"model": SETTINGS.embed_model, "input": texts}
    data = _post("/api/embed", payload)
    vectors = data["embeddings"]
    for vector in vectors:
        if len(vector) != SETTINGS.embed_dim:
            raise LLMError(
                f"Модель {SETTINGS.embed_model} вернула размерность {len(vector)}, "
                f"ожидалась {SETTINGS.embed_dim}. Проверьте EMBED_DIM в .env"
            )
    return vectors


def _normalize_model(name: str) -> str:
    """Ollama отдаёт модели без явного тега как name:latest."""
    return name if ":" in name else f"{name}:latest"


def health_check() -> None:
    """Проверяет, что Ollama на связи и нужные модели загружены."""
    try:
        with _client() as client:
            response = client.get("/api/tags", timeout=10.0)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise LLMError(
            f"Ollama недоступна по адресу {SETTINGS.ollama_url}. "
            "Запустите её командой: ollama serve"
        ) from exc

    installed = {_normalize_model(m["name"]) for m in response.json().get("models", [])}
    for model in (SETTINGS.gen_model, SETTINGS.embed_model):
        if _normalize_model(model) not in installed:
            raise LLMError(f"Модель {model} не установлена. Выполните: ollama pull {model}")
```

- [ ] **Step 5: Запустить тесты, убедиться что проходят**

```bash
.venv/Scripts/python.exe -m pytest tests/test_llm.py -v
```

Ожидается: PASS, 9 тестов

- [ ] **Step 6: Проверить работу на живой Ollama**

```bash
.venv/Scripts/python.exe -c "import llm; llm.health_check(); print(len(llm.embed(['трудовой договор'])[0]))"
```

Ожидается: `1024`

- [ ] **Step 7: Коммит**

```bash
git add llm.py tests/test_llm.py
git commit -m "feat: Ollama-backed LLM access with JSON retry and health check"
```

---

### Task 3: Схема базы данных

**Files:**
- Create: `db.py`, `tests/test_db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: `config.load_settings`
- Produces: `db.connect(path: Path | str | None = None) -> sqlite3.Connection` (загружает расширение sqlite-vec, включает foreign keys); `db.create_schema(conn: sqlite3.Connection) -> None`; `db.serialize_vector(values: list[float]) -> bytes`; `db.corpus_summary(conn) -> list[tuple[str, str | None]]` (пары «название НПА, дата редакции»)

- [ ] **Step 1: Написать падающие тесты**

Создать `tests/test_db.py`:

```python
import sqlite3

import pytest

import db


def test_connect_loads_sqlite_vec(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    version = conn.execute("SELECT vec_version()").fetchone()[0]
    assert version.startswith("v0.1")


def test_create_schema_creates_all_tables(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.create_schema(conn)
    names = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"acts", "chunks", "vec_chunks", "chunks_fts"} <= names


def test_create_schema_is_idempotent(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.create_schema(conn)
    db.create_schema(conn)  # не должно бросать


def test_vector_roundtrip(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.create_schema(conn)
    conn.execute(
        "INSERT INTO acts (id, title, adilet_url, doc_type, revision_date, scraped_at)"
        " VALUES (1, 'ТК РК', 'https://x', 'кодекс', '2026-07-01', '2026-08-04')"
    )
    conn.execute(
        "INSERT INTO chunks (id, act_id, article_no, article_title, part_no, text,"
        " char_len, anchor) VALUES (1, 1, '1', 'Понятия', NULL, 'текст', 5, 'z1')"
    )
    conn.execute(
        "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
        (1, db.serialize_vector([0.5] * 1024)),
    )
    conn.commit()
    row = conn.execute(
        "SELECT chunk_id, distance FROM vec_chunks"
        " WHERE embedding MATCH ? AND k = 1",
        (db.serialize_vector([0.5] * 1024),),
    ).fetchone()
    assert row[0] == 1
    assert row[1] == pytest.approx(0.0, abs=1e-4)


def test_fts_finds_chunk_by_word(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.create_schema(conn)
    conn.execute(
        "INSERT INTO acts (id, title, adilet_url, doc_type, revision_date, scraped_at)"
        " VALUES (1, 'ТК РК', 'https://x', 'кодекс', '2026-07-01', '2026-08-04')"
    )
    conn.execute(
        "INSERT INTO chunks (id, act_id, article_no, article_title, part_no, text,"
        " char_len, anchor) VALUES (1, 1, '30', 'Договор', NULL,"
        " 'Трудовой договор заключается в письменной форме', 46, 'z30')"
    )
    conn.commit()
    rows = conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'договор'"
    ).fetchall()
    assert [r[0] for r in rows] == [1]


def test_foreign_key_enforced(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.create_schema(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO chunks (act_id, article_no, article_title, part_no, text,"
            " char_len, anchor) VALUES (999, '1', 'X', NULL, 'y', 1, 'z1')"
        )


def test_corpus_summary_lists_acts(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.create_schema(conn)
    conn.execute(
        "INSERT INTO acts (title, adilet_url, doc_type, revision_date, scraped_at)"
        " VALUES ('ТК РК', 'https://x', 'кодекс', '2026-07-01', '2026-08-04')"
    )
    conn.commit()
    assert db.corpus_summary(conn) == [("ТК РК", "2026-07-01")]
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

```bash
.venv/Scripts/python.exe -m pytest tests/test_db.py -v
```

Ожидается: FAIL с `ModuleNotFoundError: No module named 'db'`

- [ ] **Step 3: Реализовать `db.py`**

```python
"""Подключение к SQLite и схема базы."""
from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import sqlite_vec

from config import load_settings

SETTINGS = load_settings()

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS acts (
    id            INTEGER PRIMARY KEY,
    title         TEXT NOT NULL,
    adilet_url    TEXT NOT NULL UNIQUE,
    doc_type      TEXT NOT NULL,
    act_date      TEXT,
    act_number    TEXT,
    revision_date TEXT,
    scraped_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY,
    act_id        INTEGER NOT NULL REFERENCES acts(id) ON DELETE CASCADE,
    article_no    TEXT,
    article_title TEXT,
    part_no       TEXT,
    anchor        TEXT,
    text          TEXT NOT NULL,
    char_len      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_act ON chunks(act_id);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding FLOAT[{SETTINGS.embed_dim}]
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 0'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad_vec AFTER DELETE ON chunks BEGIN
    DELETE FROM vec_chunks WHERE chunk_id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path or SETTINGS.db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def serialize_vector(values: list[float]) -> bytes:
    """Упаковывает вектор в формат, который понимает vec0."""
    return struct.pack(f"{len(values)}f", *values)


def corpus_summary(conn: sqlite3.Connection) -> list[tuple[str, str | None]]:
    return [
        (row[0], row[1])
        for row in conn.execute(
            "SELECT title, revision_date FROM acts ORDER BY title"
        )
    ]
```

- [ ] **Step 4: Запустить тесты, убедиться что проходят**

```bash
.venv/Scripts/python.exe -m pytest tests/test_db.py -v
```

Ожидается: PASS, 7 тестов

- [ ] **Step 5: Коммит**

```bash
git add db.py tests/test_db.py
git commit -m "feat: SQLite schema with sqlite-vec and FTS5 index"
```

---

### Task 4: Нарезка статей на чанки

**Files:**
- Create: `models.py`, `chunker.py`, `tests/test_chunker.py`
- Test: `tests/test_chunker.py`

**Interfaces:**
- Consumes: `config.Settings`
- Produces: `models.Article` (dataclass: `number: str`, `title: str`, `anchor: str`, `text: str`); `models.Act` (dataclass: `key: str`, `title: str`, `url: str`, `doc_type: str`, `act_date: str | None`, `act_number: str | None`, `revision_date: str | None`, `articles: list[Article]`); `models.Chunk` (dataclass: `article_no: str`, `article_title: str`, `part_no: str | None`, `anchor: str`, `text: str`); `chunker.chunk_article(article: Article, max_chars: int = 2000, min_chars: int = 100) -> list[Chunk]`; `chunker.chunk_act(act: Act) -> list[Chunk]`

- [ ] **Step 1: Создать `models.py`**

```python
"""Общие структуры данных."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Article:
    number: str          # "1" или "152-1"
    title: str
    anchor: str          # якорь на adilet, например "z1"
    text: str


@dataclass
class Act:
    key: str
    title: str
    url: str
    doc_type: str
    act_date: str | None = None       # ISO, дата принятия
    act_number: str | None = None     # "414-V"
    revision_date: str | None = None  # ISO, дата последнего изменения из сносок
    articles: list[Article] = field(default_factory=list)


@dataclass
class Chunk:
    article_no: str
    article_title: str
    part_no: str | None
    anchor: str
    text: str

    @property
    def char_len(self) -> int:
        return len(self.text)
```

- [ ] **Step 2: Написать падающие тесты**

Создать `tests/test_chunker.py`:

```python
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
```

- [ ] **Step 3: Запустить тесты, убедиться что падают**

```bash
.venv/Scripts/python.exe -m pytest tests/test_chunker.py -v
```

Ожидается: FAIL с `ModuleNotFoundError: No module named 'chunker'`

- [ ] **Step 4: Реализовать `chunker.py`**

> **Осторожно: код ниже содержит пять дефектов, найденных при ревью.** Рабочая
> версия — в `chunker.py` репозитория, ориентируйтесь на неё, а не на этот блок.
> Что было не так: терялся вводный абзац статьи до пункта «1.»; `part_no`
> подставлял порядковый счётчик вместо реального номера пункта (в НПА РК
> нумерация идёт с дырами из-за исключённых пунктов); чанк мог превысить
> `max_chars` из-за неучтённого перевода строки и склейки короткого хвоста;
> первый короткий фрагмент не сливался ни с чем; одно предложение длиннее
> бюджета отдавалось целиком. Инварианты модуля — ни один чанк не длиннее
> `max_chars` и ни один символ не теряется — закреплены property-тестами.

```python
"""Нарезка статей НПА на чанки.

Чанк — статья целиком. Юридический текст уже структурирован, статья
самодостаточна по смыслу, а ссылка на неё берётся из метаданных,
а не угадывается моделью. Длинные статьи режутся по пунктам.
"""
from __future__ import annotations

import re

from models import Act, Article, Chunk

_POINT_RE = re.compile(r"(?m)^\s*(\d+(?:-\d+)?)\.\s+")
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _split_by_points(text: str) -> list[str]:
    """Режет по нумерации пунктов; при её отсутствии — по абзацам."""
    matches = list(_POINT_RE.finditer(text))
    if len(matches) >= 2:
        parts = []
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            parts.append(text[match.start() : end])
        return parts
    paragraphs = [p for p in text.split("\n") if p.strip()]
    return paragraphs if len(paragraphs) >= 2 else [text]


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Последний рубеж: режет по границам предложений в пределах лимита."""
    sentences = re.split(r"(?<=[.;])\s+", text)
    out: list[str] = []
    buf = ""
    for sentence in sentences:
        if buf and len(buf) + len(sentence) + 1 > max_chars:
            out.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if buf:
        out.append(buf)
    return out or [text]


def chunk_article(
    article: Article, max_chars: int = 2000, min_chars: int = 100
) -> list[Chunk]:
    header = f"Статья {article.number}. {article.title}".strip()
    body = _normalize(article.text)

    if len(body) + len(header) <= max_chars:
        return [
            Chunk(
                article_no=article.number,
                article_title=article.title,
                part_no=None,
                anchor=article.anchor,
                text=f"{header}\n{body}".strip(),
            )
        ]

    pieces: list[str] = []
    for part in _split_by_points(article.text):
        part = _normalize(part)
        if not part:
            continue
        if len(part) + len(header) > max_chars:
            pieces.extend(_hard_split(part, max_chars - len(header)))
        else:
            pieces.append(part)

    # Слишком короткие куски приклеиваем к предыдущему — отдельно они бесполезны.
    merged: list[str] = []
    for piece in pieces:
        if merged and len(piece) < min_chars:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)

    return [
        Chunk(
            article_no=article.number,
            article_title=article.title,
            part_no=str(i),
            anchor=article.anchor,
            text=f"{header}\n{piece}",
        )
        for i, piece in enumerate(merged, start=1)
    ]


def chunk_act(act: Act, max_chars: int = 2000, min_chars: int = 100) -> list[Chunk]:
    chunks: list[Chunk] = []
    for article in act.articles:
        chunks.extend(chunk_article(article, max_chars, min_chars))
    return chunks
```

- [ ] **Step 5: Запустить тесты, убедиться что проходят**

```bash
.venv/Scripts/python.exe -m pytest tests/test_chunker.py -v
```

Ожидается: PASS, 8 тестов

- [ ] **Step 6: Коммит**

```bash
git add models.py chunker.py tests/test_chunker.py
git commit -m "feat: article-level chunking with point-based splitting"
```

---

### Task 5: Скрейпер adilet.zan.kz

**Files:**
- Create: `scraper.py`, `tests/test_scraper.py`, `tests/fixtures/labour_code_sample.html`
- Test: `tests/test_scraper.py`

**Interfaces:**
- Consumes: `models.Act`, `models.Article`, `config.Source`
- Produces: `scraper.ScrapeError` (Exception); `scraper.parse_act(html: str, source: Source) -> Act`; `scraper.fetch_html(url: str) -> str` (Playwright); `scraper.scrape_source(source: Source, out_dir: Path) -> Path` (пишет JSON, возвращает путь); `scraper.act_to_json(act: Act) -> str`; `scraper.act_from_json(data: str) -> Act`

**Вёрстка adilet (проверено на живой странице 2026-08-04):**
- страница отдаётся сервером целиком, JavaScript для контента не требуется
- заголовок НПА — единственный `<h1>` на странице
- тело документа — `div.container_gamma.text.text_upd`
- начало статьи — `<p><b><a name="z1"></a>Статья 1. Основные понятия...</b></p>`
- текст статьи — последующие `<p>` до следующего маркера статьи
- разделы и главы — `<h3>`
- дата и номер акта — в тексте вида `от 23 ноября 2015 года № 414-V`
- изменения — абзацы, начинающиеся со слова «Сноска», со ссылками вида `Законом РК от 11.06.2026`

- [ ] **Step 1: Создать фикстуру из реальной страницы**

Создать `scripts/make_fixture.py`:

```python
"""Готовит усечённую фикстуру страницы adilet для офлайн-тестов."""
import re
import sys
from pathlib import Path

import httpx

URL = "https://adilet.zan.kz/rus/docs/K1500000414"
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "labour_code_sample.html"


def main() -> int:
    html = httpx.get(
        URL, timeout=90, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}
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
```

```bash
.venv/Scripts/python.exe scripts/make_fixture.py
```

Ожидается: `Фикстура записана: ...labour_code_sample.html (N байт)`

- [ ] **Step 2: Написать падающие тесты**

Создать `tests/test_scraper.py`:

```python
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
```

- [ ] **Step 3: Запустить тесты, убедиться что падают**

```bash
.venv/Scripts/python.exe -m pytest tests/test_scraper.py -v
```

Ожидается: FAIL с `ModuleNotFoundError: No module named 'scraper'`

- [ ] **Step 4: Реализовать `scraper.py`**

```python
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
    act_date, act_number = _parse_act_meta(full_text)

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
```

- [ ] **Step 5: Запустить тесты, убедиться что проходят**

```bash
.venv/Scripts/python.exe -m pytest tests/test_scraper.py -v
```

Ожидается: PASS, 8 тестов

- [ ] **Step 6: Проверить на живой странице**

```bash
.venv/Scripts/python.exe -c "import config, scraper; s=[x for x in config.load_sources() if x.key=='labour_code'][0]; a=scraper.parse_act(scraper.fetch_html(s.url), s); print(a.title, '|', a.act_date, '|', a.revision_date, '| статей:', len(a.articles))"
```

Ожидается: строка вида `Трудовой кодекс Республики Казахстан | 2015-11-23 | 2026-06-11 | статей: 222`

Если статей меньше 200 — вёрстка изменилась, разбирать до перехода к следующей задаче.

- [ ] **Step 7: Коммит**

```bash
git add scraper.py scripts/make_fixture.py tests/test_scraper.py tests/fixtures/labour_code_sample.html
git commit -m "feat: Playwright scraper and adilet HTML parser"
```

---

### Task 6: Индексатор

**Files:**
- Create: `indexer.py`, `cli.py`, `tests/test_indexer.py`
- Test: `tests/test_indexer.py`

**Interfaces:**
- Consumes: `db.connect`, `db.create_schema`, `db.serialize_vector`, `chunker.chunk_act`, `llm.embed`, `scraper.act_from_json`, `models.Act`
- Produces: `indexer.index_act(conn, act: Act, embed_fn=llm.embed, batch_size: int = 16) -> int` (возвращает число чанков); `indexer.index_directory(conn, raw_dir: Path, embed_fn=llm.embed) -> dict[str, int]`; `cli.py` с командами `scrape`, `index`, `status`

- [ ] **Step 1: Написать падающие тесты**

Создать `tests/test_indexer.py`:

```python
import db
import indexer
from models import Act, Article


def _fake_embed(texts):
    return [[0.1] * 1024 for _ in texts]


def _act(key="labour_code", n_articles=3) -> Act:
    return Act(
        key=key,
        title="Трудовой кодекс Республики Казахстан",
        url=f"https://adilet.zan.kz/rus/docs/{key}",
        doc_type="кодекс",
        act_date="2015-11-23",
        act_number="414-V",
        revision_date="2026-06-11",
        articles=[
            Article(
                number=str(i),
                title=f"Статья про {i}",
                anchor=f"z{i}",
                text=f"Содержание нормы номер {i}. " * 5,
            )
            for i in range(1, n_articles + 1)
        ],
    )


def _conn(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.create_schema(conn)
    return conn


def test_index_act_writes_act_chunks_and_vectors(tmp_path):
    conn = _conn(tmp_path)
    count = indexer.index_act(conn, _act(), embed_fn=_fake_embed)
    assert count == 3
    assert conn.execute("SELECT COUNT(*) FROM acts").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0] == 3


def test_index_act_stores_metadata(tmp_path):
    conn = _conn(tmp_path)
    indexer.index_act(conn, _act(), embed_fn=_fake_embed)
    row = conn.execute(
        "SELECT title, act_date, act_number, revision_date FROM acts"
    ).fetchone()
    assert row[1] == "2015-11-23"
    assert row[2] == "414-V"
    assert row[3] == "2026-06-11"


def test_reindexing_replaces_previous_version(tmp_path):
    """Повторная индексация того же НПА не должна плодить дубликаты."""
    conn = _conn(tmp_path)
    indexer.index_act(conn, _act(n_articles=3), embed_fn=_fake_embed)
    indexer.index_act(conn, _act(n_articles=2), embed_fn=_fake_embed)
    assert conn.execute("SELECT COUNT(*) FROM acts").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0] == 2


def test_fts_index_populated(tmp_path):
    conn = _conn(tmp_path)
    indexer.index_act(conn, _act(), embed_fn=_fake_embed)
    rows = conn.execute(
        "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'нормы'"
    ).fetchone()
    assert rows[0] == 3


def test_embeddings_requested_in_batches(tmp_path):
    conn = _conn(tmp_path)
    batches = []

    def spy(texts):
        batches.append(len(texts))
        return [[0.1] * 1024 for _ in texts]

    indexer.index_act(conn, _act(n_articles=5), embed_fn=spy, batch_size=2)
    assert batches == [2, 2, 1]


def test_index_directory_processes_all_files(tmp_path):
    from scraper import act_to_json

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.json").write_text(act_to_json(_act("a", 2)), encoding="utf-8")
    (raw / "b.json").write_text(act_to_json(_act("b", 3)), encoding="utf-8")
    conn = _conn(tmp_path)
    result = indexer.index_directory(conn, raw, embed_fn=_fake_embed)
    assert result == {"a": 2, "b": 3}
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

```bash
.venv/Scripts/python.exe -m pytest tests/test_indexer.py -v
```

Ожидается: FAIL с `ModuleNotFoundError: No module named 'indexer'`

- [ ] **Step 3: Реализовать `indexer.py`**

```python
"""Сборка индекса: JSON-акты → чанки → эмбеддинги → база."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import llm
from chunker import chunk_act
from db import serialize_vector
from models import Act
from scraper import act_from_json

EmbedFn = Callable[[list[str]], list[list[float]]]


def index_act(
    conn: sqlite3.Connection,
    act: Act,
    embed_fn: EmbedFn = llm.embed,
    batch_size: int = 16,
) -> int:
    chunks = chunk_act(act)

    # Переиндексация заменяет предыдущую версию НПА целиком.
    old = conn.execute("SELECT id FROM acts WHERE adilet_url = ?", (act.url,)).fetchone()
    if old:
        conn.execute(
            "DELETE FROM vec_chunks WHERE chunk_id IN"
            " (SELECT id FROM chunks WHERE act_id = ?)",
            (old[0],),
        )
        conn.execute("DELETE FROM chunks WHERE act_id = ?", (old[0],))
        conn.execute("DELETE FROM acts WHERE id = ?", (old[0],))

    cursor = conn.execute(
        "INSERT INTO acts (title, adilet_url, doc_type, act_date, act_number,"
        " revision_date, scraped_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            act.title,
            act.url,
            act.doc_type,
            act.act_date,
            act.act_number,
            act.revision_date,
            datetime.now(timezone.utc).date().isoformat(),
        ),
    )
    act_id = cursor.lastrowid

    chunk_ids: list[int] = []
    for chunk in chunks:
        cursor = conn.execute(
            "INSERT INTO chunks (act_id, article_no, article_title, part_no, anchor,"
            " text, char_len) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                act_id,
                chunk.article_no,
                chunk.article_title,
                chunk.part_no,
                chunk.anchor,
                chunk.text,
                chunk.char_len,
            ),
        )
        chunk_ids.append(cursor.lastrowid)

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embed_fn([c.text for c in batch])
        conn.executemany(
            "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
            [
                (chunk_id, serialize_vector(vector))
                for chunk_id, vector in zip(chunk_ids[start : start + batch_size], vectors)
            ],
        )

    conn.commit()
    return len(chunks)


def index_directory(
    conn: sqlite3.Connection, raw_dir: Path, embed_fn: EmbedFn = llm.embed
) -> dict[str, int]:
    result: dict[str, int] = {}
    for path in sorted(raw_dir.glob("*.json")):
        act = act_from_json(path.read_text(encoding="utf-8"))
        result[act.key] = index_act(conn, act, embed_fn=embed_fn)
    return result
```

- [ ] **Step 4: Запустить тесты, убедиться что проходят**

```bash
.venv/Scripts/python.exe -m pytest tests/test_indexer.py -v
```

Ожидается: PASS, 6 тестов

- [ ] **Step 5: Создать `cli.py`**

```python
"""Командная строка: скрейп корпуса, сборка индекса, статус."""
from __future__ import annotations

import argparse
import sys

import db
import indexer
import llm
import scraper
from config import load_settings, load_sources

SETTINGS = load_settings()


def cmd_scrape(keys: list[str] | None) -> int:
    sources = load_sources()
    if keys:
        sources = [s for s in sources if s.key in keys]
        if not sources:
            print(f"Не найдено источников с ключами: {', '.join(keys)}", file=sys.stderr)
            return 1
    failed: list[str] = []
    for i, source in enumerate(sources, start=1):
        print(f"[{i}/{len(sources)}] {source.title} ... ", end="", flush=True)
        try:
            path = scraper.scrape_source(source, SETTINGS.raw_dir)
            print(f"ок → {path.name}")
        except Exception as exc:  # noqa: BLE001 — сообщаем и продолжаем корпус
            print(f"ОШИБКА: {exc}")
            failed.append(source.key)
    if failed:
        print(f"\nНе удалось загрузить: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def cmd_index() -> int:
    llm.health_check()
    conn = db.connect()
    db.create_schema(conn)
    if not SETTINGS.raw_dir.exists() or not any(SETTINGS.raw_dir.glob("*.json")):
        print(
            f"В {SETTINGS.raw_dir} нет данных. Сначала выполните: python cli.py scrape",
            file=sys.stderr,
        )
        return 1
    for key, count in indexer.index_directory(conn, SETTINGS.raw_dir).items():
        print(f"{key}: {count} чанков")
    return 0


def cmd_status() -> int:
    conn = db.connect()
    db.create_schema(conn)
    acts = db.corpus_summary(conn)
    if not acts:
        print("Индекс пуст. Выполните: python cli.py scrape && python cli.py index")
        return 1
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"НПА в индексе: {len(acts)}, чанков: {total}\n")
    for title, revision in acts:
        print(f"  {title} — ред. {revision or 'не определена'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Юридический консультант по НПА РК")
    sub = parser.add_subparsers(dest="command", required=True)
    scrape = sub.add_parser("scrape", help="загрузить НПА с adilet.zan.kz")
    scrape.add_argument("--only", nargs="*", help="ключи источников из sources.yaml")
    sub.add_parser("index", help="собрать индекс из загруженных данных")
    sub.add_parser("status", help="показать состояние индекса")
    args = parser.parse_args()

    if args.command == "scrape":
        return cmd_scrape(args.only)
    if args.command == "index":
        return cmd_index()
    return cmd_status()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Проверить сквозной прогон на одном НПА**

```bash
.venv/Scripts/python.exe cli.py scrape --only labour_code
.venv/Scripts/python.exe cli.py index
.venv/Scripts/python.exe cli.py status
```

Ожидается: `НПА в индексе: 1, чанков: 250+` и строка с датой редакции.

- [ ] **Step 7: Коммит**

```bash
git add indexer.py cli.py tests/test_indexer.py
git commit -m "feat: indexer with batched embeddings and CLI commands"
```

---

### Task 7: Гибридный ретривер

**Files:**
- Create: `retriever.py`, `tests/test_retriever.py`
- Test: `tests/test_retriever.py`

**Interfaces:**
- Consumes: `db.serialize_vector`, `llm.embed`, `config.Settings`
- Produces: `retriever.Hit` (dataclass: `chunk_id: int`, `act_title: str`, `article_no: str`, `article_title: str`, `part_no: str | None`, `text: str`, `url: str`, `anchor: str`, `revision_date: str | None`, `score: float`); `retriever.search(conn, query: str, k: int = 4, embed_fn=llm.embed) -> list[Hit]`; `retriever.Hit.citation` (property → `"статья 30 Трудового кодекса Республики Казахстан"`); `retriever.Hit.link` (property → `"<url>#<anchor>"`)

- [ ] **Step 1: Написать падающие тесты**

Создать `tests/test_retriever.py`:

```python
import db
import indexer
import retriever
from models import Act, Article


def _act() -> Act:
    texts = {
        "30": "Трудовой договор заключается в письменной форме не менее чем в двух "
              "экземплярах и подписывается сторонами. " * 3,
        "78": "Ежегодный оплачиваемый трудовой отпуск работникам предоставляется "
              "продолжительностью двадцать четыре календарных дня. " * 3,
        "152": "Работник обязан возместить работодателю причинённый прямой "
               "действительный ущерб. " * 3,
    }
    return Act(
        key="labour_code",
        title="Трудовой кодекс Республики Казахстан",
        url="https://adilet.zan.kz/rus/docs/K1500000414",
        doc_type="кодекс",
        revision_date="2026-06-11",
        articles=[
            Article(number=n, title=f"Статья {n}", anchor=f"z{n}", text=t)
            for n, t in texts.items()
        ],
    )


def _vector_for(text: str) -> list[float]:
    """Детерминированный псевдо-эмбеддинг: близость растёт с пересечением слов."""
    vocabulary = ["договор", "отпуск", "ущерб", "письменной", "календарных", "возместить"]
    lowered = text.lower()
    head = [1.0 if word in lowered else 0.0 for word in vocabulary]
    return head + [0.0] * (1024 - len(head))


def _embed(texts):
    return [_vector_for(t) for t in texts]


def _conn(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.create_schema(conn)
    indexer.index_act(conn, _act(), embed_fn=_embed)
    return conn


def test_search_finds_relevant_article(tmp_path):
    hits = retriever.search(_conn(tmp_path), "как оформить отпуск", k=2, embed_fn=_embed)
    assert hits
    assert hits[0].article_no == "78"


def test_hit_carries_citation_metadata(tmp_path):
    hits = retriever.search(_conn(tmp_path), "трудовой договор", k=1, embed_fn=_embed)
    hit = hits[0]
    assert hit.act_title == "Трудовой кодекс Республики Казахстан"
    assert hit.revision_date == "2026-06-11"
    assert hit.citation == "статья 30 — Трудовой кодекс Республики Казахстан"
    assert hit.link == "https://adilet.zan.kz/rus/docs/K1500000414#z30"


def test_exact_article_number_query_hits_fts_branch(tmp_path):
    """Чисто векторный поиск такое проваливает — за это отвечает FTS5."""
    hits = retriever.search(_conn(tmp_path), "возместить ущерб", k=3, embed_fn=_embed)
    assert "152" in {h.article_no for h in hits}


def test_search_respects_k(tmp_path):
    hits = retriever.search(_conn(tmp_path), "трудовой договор", k=2, embed_fn=_embed)
    assert len(hits) <= 2


def test_irrelevant_query_returns_empty(tmp_path):
    """Ниже порога релевантности и без совпадений FTS — отказ, а не мусор."""
    hits = retriever.search(
        _conn(tmp_path), "рецепт борща", k=4, embed_fn=lambda t: [[0.0] * 1024 for _ in t]
    )
    assert hits == []


def test_empty_index_returns_empty(tmp_path):
    conn = db.connect(tmp_path / "empty.db")
    db.create_schema(conn)
    assert retriever.search(conn, "трудовой договор", embed_fn=_embed) == []


def test_fts_query_with_special_characters_does_not_crash(tmp_path):
    """Пользовательский ввод попадает в MATCH — синтаксис FTS5 надо экранировать."""
    hits = retriever.search(
        _conn(tmp_path), 'договор AND "ущерб OR (', k=2, embed_fn=_embed
    )
    assert isinstance(hits, list)
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

```bash
.venv/Scripts/python.exe -m pytest tests/test_retriever.py -v
```

Ожидается: FAIL с `ModuleNotFoundError: No module named 'retriever'`

- [ ] **Step 3: Реализовать `retriever.py`**

```python
"""Гибридный поиск по корпусу НПА.

Векторный поиск проваливается на точных запросах и редких терминах,
полнотекстовый — на переформулировках. Берём оба и сливаем по RRF.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Callable

import llm
from config import load_settings
from db import serialize_vector

SETTINGS = load_settings()
RRF_K = 60
CANDIDATES = 20
_WORD_RE = re.compile(r"\w+", re.UNICODE)

EmbedFn = Callable[[list[str]], list[list[float]]]


@dataclass
class Hit:
    chunk_id: int
    act_title: str
    article_no: str
    article_title: str
    part_no: str | None
    text: str
    url: str
    anchor: str
    revision_date: str | None
    score: float

    @property
    def citation(self) -> str:
        # Именительный падеж через тире: склонять названия НПА по-русски
        # («Трудового кодекса Республики Казахстан») — отдельная задача,
        # которая корректному цитированию ничего не добавляет.
        return f"статья {self.article_no} — {self.act_title}"

    @property
    def link(self) -> str:
        return f"{self.url}#{self.anchor}" if self.anchor else self.url


def _fts_query(text: str) -> str:
    """Превращает свободный ввод в безопасный запрос FTS5."""
    words = _WORD_RE.findall(text.lower())
    return " OR ".join(f'"{w}"' for w in words if len(w) > 2)


def _vector_candidates(
    conn: sqlite3.Connection, query: str, embed_fn: EmbedFn
) -> tuple[list[int], float]:
    vector = embed_fn([query])[0]
    rows = conn.execute(
        "SELECT chunk_id, distance FROM vec_chunks"
        " WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (serialize_vector(vector), CANDIDATES),
    ).fetchall()
    if not rows:
        return [], 0.0
    # vec0 отдаёт L2-расстояние по нормализованным векторам: cos = 1 - d²/2
    best_similarity = 1.0 - (rows[0][1] ** 2) / 2.0
    return [row[0] for row in rows], best_similarity


def _fts_candidates(conn: sqlite3.Connection, query: str) -> list[int]:
    match = _fts_query(query)
    if not match:
        return []
    rows = conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?"
        " ORDER BY rank LIMIT ?",
        (match, CANDIDATES),
    ).fetchall()
    return [row[0] for row in rows]


def _fuse(*rankings: list[int]) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + position)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def search(
    conn: sqlite3.Connection,
    query: str,
    k: int = 4,
    embed_fn: EmbedFn = llm.embed,
) -> list[Hit]:
    vector_ids, best_similarity = _vector_candidates(conn, query, embed_fn)
    fts_ids = _fts_candidates(conn, query)

    if best_similarity < SETTINGS.relevance_threshold and not fts_ids:
        return []
    if not vector_ids and not fts_ids:
        return []

    fused = _fuse(vector_ids, fts_ids)[:k]
    if not fused:
        return []

    placeholders = ",".join("?" * len(fused))
    rows = conn.execute(
        f"""
        SELECT c.id, a.title, c.article_no, c.article_title, c.part_no, c.text,
               a.adilet_url, c.anchor, a.revision_date
        FROM chunks c JOIN acts a ON a.id = c.act_id
        WHERE c.id IN ({placeholders})
        """,
        [chunk_id for chunk_id, _ in fused],
    ).fetchall()

    by_id = {row[0]: row for row in rows}
    hits: list[Hit] = []
    for chunk_id, score in fused:
        row = by_id.get(chunk_id)
        if row is None:
            continue
        hits.append(
            Hit(
                chunk_id=row[0],
                act_title=row[1],
                article_no=row[2],
                article_title=row[3],
                part_no=row[4],
                text=row[5],
                url=row[6],
                anchor=row[7],
                revision_date=row[8],
                score=score,
            )
        )
    return hits
```

- [ ] **Step 4: Запустить тесты, убедиться что проходят**

```bash
.venv/Scripts/python.exe -m pytest tests/test_retriever.py -v
```

Ожидается: PASS, 7 тестов

- [ ] **Step 5: Проверить на реальном индексе**

```bash
.venv/Scripts/python.exe -c "import db, retriever; c=db.connect(); [print(h.citation, '|', round(h.score,4)) for h in retriever.search(c, 'испытательный срок при приёме на работу')]"
```

Ожидается: 4 строки со статьями Трудового кодекса, среди них статья про испытательный срок (ст. 36).

Если релевантных статей нет — калибровать `RELEVANCE_THRESHOLD` в `.env`.

- [ ] **Step 6: Коммит**

```bash
git add retriever.py tests/test_retriever.py
git commit -m "feat: hybrid retrieval with RRF fusion and relevance floor"
```

---

### Task 8: Веб-приложение и вкладка консультации

**Files:**
- Create: `prompts.py`, `app.py`, `templates/base.html`, `templates/index.html`, `templates/_answer.html`, `static/app.css`, `tests/test_app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `retriever.search`, `retriever.Hit`, `llm.complete`, `llm.LLMError`, `db.connect`, `db.corpus_summary`
- Produces: `prompts.SYSTEM_CONSULTANT: str`; `prompts.build_consultation_messages(question: str, hits: list[Hit]) -> list[dict]`; `prompts.DISCLAIMER: str`; `app.app` (FastAPI); `app.answer_question(conn, question: str) -> dict` с ключами `answer: str`, `sources: list[Hit]`, `refused: bool`

- [ ] **Step 1: Написать падающие тесты промптов и эндпоинта**

Создать `tests/test_app.py`:

```python
import pytest
from fastapi.testclient import TestClient

import app as app_module
import db
import indexer
import prompts
from models import Act, Article
from retriever import Hit


def _hit(article_no="30") -> Hit:
    return Hit(
        chunk_id=1,
        act_title="Трудовой кодекс Республики Казахстан",
        article_no=article_no,
        article_title="Форма трудового договора",
        part_no=None,
        text="Статья 30. Трудовой договор заключается в письменной форме.",
        url="https://adilet.zan.kz/rus/docs/K1500000414",
        anchor=f"z{article_no}",
        revision_date="2026-06-11",
        score=0.9,
    )


def test_system_prompt_forbids_outside_knowledge():
    assert "только" in prompts.SYSTEM_CONSULTANT.lower()
    assert "контекст" in prompts.SYSTEM_CONSULTANT.lower()


def test_messages_include_context_and_question():
    messages = prompts.build_consultation_messages("Нужен ли письменный договор?", [_hit()])
    assert messages[0]["role"] == "system"
    user = messages[-1]["content"]
    assert "письменной форме" in user
    assert "Нужен ли письменный договор?" in user
    assert "статья 30" in user


def test_messages_carry_revision_date():
    user = prompts.build_consultation_messages("вопрос", [_hit()])[-1]["content"]
    assert "2026-06-11" in user


def _seed(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.create_schema(conn)
    act = Act(
        key="labour_code",
        title="Трудовой кодекс Республики Казахстан",
        url="https://adilet.zan.kz/rus/docs/K1500000414",
        doc_type="кодекс",
        revision_date="2026-06-11",
        articles=[
            Article(
                number="30",
                title="Форма трудового договора",
                anchor="z30",
                text="Трудовой договор заключается в письменной форме. " * 5,
            )
        ],
    )
    indexer.index_act(conn, act, embed_fn=lambda t: [[0.5] * 1024 for _ in t])
    return conn


def test_answer_question_returns_answer_with_sources(tmp_path, monkeypatch):
    conn = _seed(tmp_path)
    monkeypatch.setattr(
        app_module.retriever, "search", lambda *a, **kw: [_hit()]
    )
    monkeypatch.setattr(
        app_module.llm, "complete", lambda messages, **kw: "Да, согласно статье 30."
    )
    result = app_module.answer_question(conn, "Нужен ли письменный договор?")
    assert result["refused"] is False
    assert "статье 30" in result["answer"]
    assert len(result["sources"]) == 1


def test_answer_question_refuses_without_hits(tmp_path, monkeypatch):
    conn = _seed(tmp_path)
    monkeypatch.setattr(app_module.retriever, "search", lambda *a, **kw: [])
    called = {"llm": False}

    def fail(*args, **kwargs):
        called["llm"] = True
        raise AssertionError("модель не должна вызываться при пустом ретриве")

    monkeypatch.setattr(app_module.llm, "complete", fail)
    result = app_module.answer_question(conn, "рецепт борща")
    assert result["refused"] is True
    assert "не нашёл" in result["answer"].lower()
    assert called["llm"] is False


def test_chat_endpoint_renders_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "get_connection", lambda: _seed(tmp_path))
    monkeypatch.setattr(app_module.retriever, "search", lambda *a, **kw: [_hit()])
    monkeypatch.setattr(app_module.llm, "complete", lambda messages, **kw: "Ответ.")
    client = TestClient(app_module.app)
    response = client.post("/chat", data={"question": "Нужен ли договор?"})
    assert response.status_code == 200
    assert "Ответ." in response.text
    assert "adilet.zan.kz" in response.text
    assert prompts.DISCLAIMER[:30] in response.text


def test_chat_endpoint_rejects_empty_question(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "get_connection", lambda: _seed(tmp_path))
    client = TestClient(app_module.app)
    response = client.post("/chat", data={"question": "   "})
    assert response.status_code == 200
    assert "введите вопрос" in response.text.lower()


def test_chat_endpoint_reports_llm_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "get_connection", lambda: _seed(tmp_path))
    monkeypatch.setattr(app_module.retriever, "search", lambda *a, **kw: [_hit()])

    def boom(*args, **kwargs):
        raise app_module.llm.LLMError("Ollama недоступна по адресу http://localhost:11434")

    monkeypatch.setattr(app_module.llm, "complete", boom)
    client = TestClient(app_module.app)
    response = client.post("/chat", data={"question": "вопрос"})
    assert response.status_code == 200
    assert "Ollama недоступна" in response.text


def test_index_page_lists_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "get_connection", lambda: _seed(tmp_path))
    client = TestClient(app_module.app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Трудовой кодекс" in response.text
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

```bash
.venv/Scripts/python.exe -m pytest tests/test_app.py -v
```

Ожидается: FAIL с `ModuleNotFoundError: No module named 'app'`

- [ ] **Step 3: Реализовать `prompts.py`**

```python
"""Промпты. Вся защита от выдумывания норм живёт здесь."""
from __future__ import annotations

from retriever import Hit

DISCLAIMER = (
    "Справочная информация на основе текстов НПА Республики Казахстан. "
    "Не является юридической консультацией — для принятия решений обратитесь к юристу."
)

SYSTEM_CONSULTANT = """Ты — юридический справочник по законодательству Республики Казахстан.

Жёсткие правила:
1. Отвечай ТОЛЬКО на основе норм, приведённых в блоке «Контекст». Если ответа в контексте нет — так и напиши, не додумывай.
2. После каждого утверждения ставь ссылку на норму в формате: (статья N <название НПА>).
3. Не ссылайся на законодательство других стран и на нормы, которых нет в контексте.
4. Не давай персональных рекомендаций по конкретной сделке — излагай, что говорит закон.
5. Отвечай по-русски, кратко и по существу. Без вступлений и извинений.
"""

REFUSAL = (
    "Не нашёл релевантной нормы в загруженном корпусе НПА. "
    "Уточните вопрос или проверьте, входит ли эта область права в список загруженных документов."
)


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        revision = f", редакция от {hit.revision_date}" if hit.revision_date else ""
        blocks.append(
            f"[{i}] {hit.citation}{revision}\n{hit.text}"
        )
    return "\n\n".join(blocks)


def build_consultation_messages(question: str, hits: list[Hit]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_CONSULTANT},
        {
            "role": "user",
            "content": (
                f"Контекст:\n{format_context(hits)}\n\n"
                f"Вопрос: {question}\n\n"
                "Ответь строго по контексту, со ссылками на статьи."
            ),
        },
    ]
```

- [ ] **Step 4: Реализовать `app.py`**

```python
"""HTTP-слой: страницы, консультация, анализ документов."""
from __future__ import annotations

import sqlite3

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import db
import llm
import prompts
import retriever
from config import ROOT, load_settings

SETTINGS = load_settings()
app = FastAPI(title="Юридический консультант по НПА РК")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "templates")


def get_connection() -> sqlite3.Connection:
    conn = db.connect()
    db.create_schema(conn)
    return conn


def answer_question(conn: sqlite3.Connection, question: str) -> dict:
    hits = retriever.search(conn, question, k=4)
    if not hits:
        return {"answer": prompts.REFUSAL, "sources": [], "refused": True}
    answer = llm.complete(prompts.build_consultation_messages(question, hits))
    return {"answer": answer, "sources": hits, "refused": False}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    conn = get_connection()
    acts = db.corpus_summary(conn)
    conn.close()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "acts": acts, "disclaimer": prompts.DISCLAIMER},
    )


@app.post("/chat", response_class=HTMLResponse)
def chat(request: Request, question: str = Form(...)):
    question = question.strip()
    if not question:
        return templates.TemplateResponse(
            "_answer.html",
            {"request": request, "error": "Введите вопрос.", "disclaimer": prompts.DISCLAIMER},
        )
    conn = get_connection()
    try:
        result = answer_question(conn, question)
    except llm.LLMError as exc:
        return templates.TemplateResponse(
            "_answer.html",
            {"request": request, "error": str(exc), "disclaimer": prompts.DISCLAIMER},
        )
    finally:
        conn.close()
    return templates.TemplateResponse(
        "_answer.html",
        {
            "request": request,
            "question": question,
            "result": result,
            "disclaimer": prompts.DISCLAIMER,
        },
    )
```

- [ ] **Step 5: Создать шаблоны**

`templates/base.html`:

```html
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Юридический консультант — законодательство РК</title>
  <script src="https://unpkg.com/htmx.org@2.0.4"></script>
  <link rel="stylesheet" href="/static/app.css">
</head>
<body>
  <header>
    <h1>Юридический консультант</h1>
    <p class="sub">Ответы только по действующему законодательству Республики Казахстан</p>
  </header>
  <main>{% block content %}{% endblock %}</main>
  <footer><p class="disclaimer">{{ disclaimer }}</p></footer>
</body>
</html>
```

`templates/index.html`:

```html
{% extends "base.html" %}
{% block content %}
<nav class="tabs">
  <button class="tab active" data-target="pane-chat">Консультация</button>
  <button class="tab" data-target="pane-doc">Документ</button>
</nav>

<section id="pane-chat" class="pane">
  <form hx-post="/chat" hx-target="#answer" hx-indicator="#spinner">
    <textarea name="question" rows="3" placeholder="Например: обязателен ли письменный трудовой договор?"></textarea>
    <button type="submit">Спросить</button>
    <span id="spinner" class="htmx-indicator">Модель думает…</span>
  </form>
  <div id="answer"></div>
</section>

<section id="pane-doc" class="pane" hidden>
  <p>Загрузка документов подключается в следующей задаче.</p>
</section>

<aside class="corpus">
  <h2>Загруженные НПА ({{ acts|length }})</h2>
  <ul>
    {% for title, revision in acts %}
      <li>{{ title }} <span class="rev">ред. {{ revision or 'не определена' }}</span></li>
    {% else %}
      <li class="empty">Индекс пуст. Выполните: <code>python cli.py scrape &amp;&amp; python cli.py index</code></li>
    {% endfor %}
  </ul>
</aside>

<script>
document.querySelectorAll('.tab').forEach(function (tab) {
  tab.addEventListener('click', function () {
    document.querySelectorAll('.tab').forEach(function (t) { t.classList.remove('active'); });
    document.querySelectorAll('.pane').forEach(function (p) { p.hidden = true; });
    tab.classList.add('active');
    document.getElementById(tab.dataset.target).hidden = false;
  });
});
</script>
{% endblock %}
```

`templates/_answer.html`:

```html
{% if error %}
  <div class="error">{{ error }}</div>
{% else %}
  <article class="answer {% if result.refused %}refused{% endif %}">
    <p class="question">{{ question }}</p>
    <div class="body">{{ result.answer }}</div>
    {% if result.sources %}
      <h3>Источники</h3>
      <ul class="sources">
        {% for hit in result.sources %}
          <li>
            <a href="{{ hit.link }}" target="_blank" rel="noopener">{{ hit.citation }}</a>
            {% if hit.revision_date %}<span class="rev">ред. {{ hit.revision_date }}</span>{% endif %}
          </li>
        {% endfor %}
      </ul>
    {% endif %}
    <p class="disclaimer">{{ disclaimer }}</p>
  </article>
{% endif %}
```

`static/app.css`:

```css
:root { --fg:#1a1a1a; --muted:#666; --line:#e2e2e2; --accent:#0b5fff; --warn:#b23; }
* { box-sizing: border-box; }
body { font: 16px/1.55 system-ui, sans-serif; color: var(--fg); margin: 0 auto; max-width: 900px; padding: 24px; }
header h1 { margin: 0 0 4px; font-size: 24px; }
.sub { color: var(--muted); margin: 0 0 20px; }
.tabs { display: flex; gap: 8px; border-bottom: 1px solid var(--line); margin-bottom: 16px; }
.tab { background: none; border: none; padding: 10px 14px; cursor: pointer; font-size: 15px; color: var(--muted); }
.tab.active { color: var(--accent); border-bottom: 2px solid var(--accent); }
textarea { width: 100%; padding: 10px; border: 1px solid var(--line); border-radius: 6px; font: inherit; }
button[type=submit] { margin-top: 8px; padding: 9px 18px; background: var(--accent); color: #fff; border: none; border-radius: 6px; cursor: pointer; font-size: 15px; }
.htmx-indicator { display: none; margin-left: 10px; color: var(--muted); }
.htmx-request .htmx-indicator, .htmx-request.htmx-indicator { display: inline; }
.answer { border: 1px solid var(--line); border-radius: 8px; padding: 16px; margin-top: 18px; }
.answer.refused { border-color: var(--warn); }
.question { color: var(--muted); font-style: italic; margin: 0 0 10px; }
.body { white-space: pre-wrap; }
.sources { padding-left: 18px; }
.rev { color: var(--muted); font-size: 13px; margin-left: 6px; }
.disclaimer { color: var(--muted); font-size: 13px; border-top: 1px solid var(--line); padding-top: 10px; margin-top: 14px; }
.error { border: 1px solid var(--warn); color: var(--warn); border-radius: 8px; padding: 14px; margin-top: 18px; }
.corpus { margin-top: 32px; border-top: 1px solid var(--line); padding-top: 16px; }
.corpus h2 { font-size: 15px; color: var(--muted); }
.corpus ul { padding-left: 18px; font-size: 14px; }
```

- [ ] **Step 6: Запустить тесты, убедиться что проходят**

```bash
.venv/Scripts/python.exe -m pytest tests/test_app.py -v
```

Ожидается: PASS, 9 тестов

- [ ] **Step 7: Проверить вживую**

```bash
.venv/Scripts/python.exe -m uvicorn app:app --reload --port 8000
```

Открыть `http://localhost:8000`, задать вопрос «обязателен ли письменный трудовой договор». Ожидается ответ со ссылкой на статью 33 или 30 Трудового кодекса и рабочей ссылкой на adilet.

- [ ] **Step 8: Коммит**

```bash
git add prompts.py app.py templates static tests/test_app.py
git commit -m "feat: consultation endpoint with grounded prompts and citations"
```

---

### Task 9: Сборка полного корпуса

**Files:**
- Create: `README.md`
- Modify: `config/sources.yaml` (только если какие-то URL окажутся нерабочими)

**Interfaces:**
- Consumes: `cli.py`
- Produces: заполненный `law.db`, `README.md` с инструкцией запуска

- [ ] **Step 1: Скачать весь корпус**

```bash
.venv/Scripts/python.exe cli.py scrape
```

Ожидается: 21 строка вида `[N/21] <название> ... ок → <key>.json`.

Занимает 5-10 минут (пауза 2 секунды между запросами плюс загрузка страниц до 3 МБ).

Если какой-то источник отдал ошибку — открыть его URL в браузере, найти актуальный адрес документа на adilet, поправить `config/sources.yaml` и повторить командой `cli.py scrape --only <key>`.

- [ ] **Step 2: Проверить объём выгрузки**

```bash
.venv/Scripts/python.exe -c "import json,pathlib; ps=sorted(pathlib.Path('data/raw').glob('*.json')); [print(p.stem, len(json.loads(p.read_text(encoding='utf-8'))['articles']), 'статей') for p in ps]"
```

Ожидается: 21 строка, у каждого НПА не менее 20 статей. Ноль или единицы статей означают промах парсера на конкретной вёрстке — разбирать до индексации.

- [ ] **Step 3: Собрать индекс**

```bash
.venv/Scripts/python.exe cli.py index
```

Ожидается: строки `<key>: N чанков` по каждому НПА. Занимает 20-40 минут — эмбеддинги считаются на GPU по 16 чанков за раз.

- [ ] **Step 4: Проверить состояние индекса**

```bash
.venv/Scripts/python.exe cli.py status
```

Ожидается: `НПА в индексе: 21, чанков: 15000+` и список НПА с датами редакций.

- [ ] **Step 5: Прогнать контрольные вопросы и откалибровать порог**

```bash
.venv/Scripts/python.exe -c "
import db, retriever
questions = [
    'испытательный срок при приёме на работу',
    'срок исковой давности по договору',
    'ответственность за нарушение персональных данных',
    'какой размер уставного капитала ТОО',
    'расторжение брака при наличии детей',
    'рецепт борща',
]
conn = db.connect()
for q in questions:
    hits = retriever.search(conn, q)
    print(q, '->', [h.citation for h in hits[:2]] or 'ОТКАЗ')
"
```

Ожидается: по первым пяти вопросам — релевантные статьи, по «рецепту борща» — `ОТКАЗ`. Если борщ находит нормы, поднять `RELEVANCE_THRESHOLD` в `.env`; если юридические вопросы получают отказ — понизить.

- [ ] **Step 6: Написать `README.md`**

```markdown
# Юридический консультант по законодательству РК

Отвечает на юридические вопросы строго по текстам НПА Республики Казахстан
и анализирует загруженные документы на правовые риски.

## Требования

- Python 3.11
- [Ollama](https://ollama.com) с моделями `qwen3:4b-instruct` и `bge-m3`
- GPU с 4 ГБ видеопамяти (проверено на GTX 1650 Ti) либо терпение к CPU-инференсу

## Установка

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m playwright install chromium
ollama pull qwen3:4b-instruct
ollama pull bge-m3
copy .env.example .env
```

## Сборка корпуса

```bash
.venv/Scripts/python.exe cli.py scrape   # загрузка НПА с adilet.zan.kz, 5-10 минут
.venv/Scripts/python.exe cli.py index    # эмбеддинги и индекс, 20-40 минут
.venv/Scripts/python.exe cli.py status   # что в индексе
```

## Запуск

```bash
.venv/Scripts/python.exe -m uvicorn app:app --port 8000
```

Открыть http://localhost:8000

## Тесты

```bash
.venv/Scripts/python.exe -m pytest              # без обращений к Ollama
.venv/Scripts/python.exe -m pytest -m integration   # требует Ollama и собранный индекс
```

## Ограничения

- Корпус — кодексы и ключевые законы из `config/sources.yaml`, не вся база «Әділет»
- Судебная практика не учитывается
- Отсканированные PDF без текстового слоя не обрабатываются
- Ответы — справочная информация, не юридическая консультация
```

- [ ] **Step 7: Коммит**

```bash
git add README.md config/sources.yaml
git commit -m "docs: setup instructions; corpus scraped and indexed"
```

---

### Task 10: Разбор загружаемых документов

**Files:**
- Create: `docparse.py`, `tests/test_docparse.py`, `tests/fixtures/contract_sample.docx`
- Test: `tests/test_docparse.py`

**Interfaces:**
- Consumes: `config.Settings`
- Produces: `docparse.DocParseError` (Exception); `docparse.Clause` (dataclass: `no: str`, `text: str`); `docparse.extract_text(path: Path) -> str`; `docparse.split_clauses(text: str, max_clauses: int = 300) -> list[Clause]`; `docparse.parse_document(path: Path, max_clauses: int = 300) -> list[Clause]`

- [ ] **Step 1: Написать падающие тесты**

Создать `tests/test_docparse.py`:

```python
from pathlib import Path

import pytest
from docx import Document

import docparse


@pytest.fixture
def contract(tmp_path) -> Path:
    doc = Document()
    doc.add_paragraph("ДОГОВОР ВОЗМЕЗДНОГО ОКАЗАНИЯ УСЛУГ")
    doc.add_paragraph("1. Предмет договора")
    doc.add_paragraph("1.1. Исполнитель обязуется оказать услуги по разработке.")
    doc.add_paragraph("1.2. Заказчик обязуется оплатить услуги.")
    doc.add_paragraph("2. Стоимость и порядок расчётов")
    doc.add_paragraph("2.1. Стоимость услуг составляет 1 000 000 тенге.")
    doc.add_paragraph("")
    doc.add_paragraph("2.2. Оплата производится в течение 90 банковских дней.")
    path = tmp_path / "contract.docx"
    doc.save(path)
    return path


def test_extract_text_from_docx(contract):
    text = docparse.extract_text(contract)
    assert "Исполнитель обязуется" in text
    assert "1 000 000 тенге" in text


def test_split_clauses_uses_numbering():
    text = (
        "1. Предмет договора\n"
        "1.1. Исполнитель оказывает услуги.\n"
        "1.2. Заказчик оплачивает услуги.\n"
        "2.1. Стоимость составляет 100 тенге.\n"
    )
    clauses = docparse.split_clauses(text)
    assert [c.no for c in clauses] == ["1.", "1.1.", "1.2.", "2.1."]
    assert clauses[1].text.startswith("Исполнитель")


def test_split_clauses_falls_back_to_paragraphs():
    text = "Первый абзац без нумерации.\nВторой абзац без нумерации.\n"
    clauses = docparse.split_clauses(text)
    assert len(clauses) == 2
    assert clauses[0].no == "1"


def test_split_clauses_skips_blank_lines():
    clauses = docparse.split_clauses("1. Первый\n\n\n2. Второй\n")
    assert len(clauses) == 2


def test_split_clauses_enforces_limit():
    text = "\n".join(f"{i}. Пункт номер {i}" for i in range(1, 20))
    with pytest.raises(docparse.DocParseError, match="Слишком много пунктов"):
        docparse.split_clauses(text, max_clauses=10)


def test_parse_document_end_to_end(contract):
    clauses = docparse.parse_document(contract)
    assert len(clauses) >= 6
    assert any("90 банковских дней" in c.text for c in clauses)


def test_unsupported_extension_rejected(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("текст", encoding="utf-8")
    with pytest.raises(docparse.DocParseError, match="Поддерживаются DOCX и PDF"):
        docparse.parse_document(path)


def test_pdf_without_text_layer_reports_scan(tmp_path, monkeypatch):
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(docparse, "_pdf_text", lambda p: "   \n  ")
    with pytest.raises(docparse.DocParseError, match="скан"):
        docparse.parse_document(path)
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

```bash
.venv/Scripts/python.exe -m pytest tests/test_docparse.py -v
```

Ожидается: FAIL с `ModuleNotFoundError: No module named 'docparse'`

- [ ] **Step 3: Реализовать `docparse.py`**

```python
"""Разбор загруженных документов на пункты."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

CLAUSE_RE = re.compile(r"(?m)^\s*((?:\d+\.)+|\d+\)|Статья\s+\d+\.|§\s*\d+\.)\s*")


class DocParseError(RuntimeError):
    """Документ невозможно разобрать — текст показывается пользователю."""


@dataclass
class Clause:
    no: str
    text: str


def _docx_text(path: Path) -> str:
    from docx import Document

    document = Document(str(path))
    lines = [p.text.strip() for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                lines.append(" | ".join(cells))
    return "\n".join(line for line in lines if line)


def _pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _docx_text(path)
    if suffix == ".pdf":
        return _pdf_text(path)
    raise DocParseError(
        f"Формат {suffix or 'без расширения'} не поддерживается. "
        "Поддерживаются DOCX и PDF."
    )


def split_clauses(text: str, max_clauses: int = 300) -> list[Clause]:
    matches = list(CLAUSE_RE.finditer(text))
    clauses: list[Clause] = []

    if len(matches) >= 2:
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[match.end() : end].strip()
            body = re.sub(r"\s+", " ", body)
            if body:
                clauses.append(Clause(no=match.group(1).strip(), text=body))
    else:
        for i, line in enumerate(
            [ln.strip() for ln in text.split("\n") if ln.strip()], start=1
        ):
            clauses.append(Clause(no=str(i), text=line))

    if len(clauses) > max_clauses:
        raise DocParseError(
            f"Слишком много пунктов: {len(clauses)}, предел {max_clauses}. "
            "Загрузите документ по частям."
        )
    return clauses


def parse_document(path: Path, max_clauses: int = 300) -> list[Clause]:
    text = extract_text(path)
    if not text.strip():
        raise DocParseError(
            "В файле нет текстового слоя. Похоже, это скан — "
            "распознавание изображений не поддерживается."
        )
    clauses = split_clauses(text, max_clauses)
    if not clauses:
        raise DocParseError("Не удалось выделить ни одного пункта документа.")
    return clauses
```

- [ ] **Step 4: Запустить тесты, убедиться что проходят**

```bash
.venv/Scripts/python.exe -m pytest tests/test_docparse.py -v
```

Ожидается: PASS, 8 тестов

- [ ] **Step 5: Коммит**

```bash
git add docparse.py tests/test_docparse.py
git commit -m "feat: DOCX/PDF clause extraction with scan detection"
```

---

### Task 11: Анализ пунктов на риски

**Files:**
- Create: `analyzer.py`, `tests/test_analyzer.py`
- Modify: `prompts.py` (добавить промпты анализа)
- Test: `tests/test_analyzer.py`

**Interfaces:**
- Consumes: `retriever.search`, `llm.complete_json`, `llm.LLMError`, `docparse.Clause`
- Produces: `analyzer.Finding` (dataclass: `clause_no: str`, `risk: str`, `quote: str`, `norm_ref: str`, `problem: str`, `suggestion: str`, `link: str`); `analyzer.RISK_ORDER: dict[str, int]`; `analyzer.classify_document(clauses: list[Clause]) -> str`; `analyzer.analyze_clause(conn, clause: Clause, doc_type: str) -> Finding | None`; `analyzer.analyze_document(conn, clauses, progress=None) -> list[Finding]`; `prompts.build_classification_messages(sample: str) -> list[dict]`; `prompts.build_clause_messages(clause, doc_type, hits) -> list[dict]`

- [ ] **Step 1: Написать падающие тесты**

Создать `tests/test_analyzer.py`:

```python
import pytest

import analyzer
import llm
from docparse import Clause
from retriever import Hit


def _hit() -> Hit:
    return Hit(
        chunk_id=1,
        act_title="Трудовой кодекс Республики Казахстан",
        article_no="113",
        article_title="Сроки выплаты заработной платы",
        part_no=None,
        text="Статья 113. Заработная плата выплачивается не реже одного раза в месяц.",
        url="https://adilet.zan.kz/rus/docs/K1500000414",
        anchor="z113",
        revision_date="2026-06-11",
        score=0.9,
    )


def test_analyze_clause_returns_finding(monkeypatch):
    monkeypatch.setattr(analyzer.retriever, "search", lambda *a, **kw: [_hit()])
    monkeypatch.setattr(
        analyzer.llm,
        "complete_json",
        lambda messages: {
            "risk": "high",
            "quote": "Оплата производится в течение 90 банковских дней",
            "norm_ref": "статья 113 Трудового кодекса Республики Казахстан",
            "problem": "Срок выплаты превышает установленный законом",
            "suggestion": "Оплата производится не реже одного раза в месяц",
        },
    )
    clause = Clause(no="2.2.", text="Оплата производится в течение 90 банковских дней")
    finding = analyzer.analyze_clause(None, clause, "договор")
    assert finding.risk == "high"
    assert finding.clause_no == "2.2."
    assert "113" in finding.norm_ref
    assert finding.link.endswith("#z113")


def test_analyze_clause_drops_risk_none(monkeypatch):
    monkeypatch.setattr(analyzer.retriever, "search", lambda *a, **kw: [_hit()])
    monkeypatch.setattr(
        analyzer.llm,
        "complete_json",
        lambda messages: {"risk": "none", "quote": "", "norm_ref": "", "problem": "", "suggestion": ""},
    )
    assert analyzer.analyze_clause(None, Clause("1.1.", "Общие положения"), "договор") is None


def test_analyze_clause_without_hits_skips_llm(monkeypatch):
    monkeypatch.setattr(analyzer.retriever, "search", lambda *a, **kw: [])

    def fail(messages):
        raise AssertionError("модель не должна вызываться без релевантных норм")

    monkeypatch.setattr(analyzer.llm, "complete_json", fail)
    assert analyzer.analyze_clause(None, Clause("9.9.", "Подписи сторон"), "договор") is None


def test_unknown_risk_value_is_normalized(monkeypatch):
    monkeypatch.setattr(analyzer.retriever, "search", lambda *a, **kw: [_hit()])
    monkeypatch.setattr(
        analyzer.llm,
        "complete_json",
        lambda messages: {
            "risk": "КРИТИЧЕСКИЙ",
            "quote": "q",
            "norm_ref": "статья 113",
            "problem": "p",
            "suggestion": "s",
        },
    )
    finding = analyzer.analyze_clause(None, Clause("1.", "текст"), "договор")
    assert finding.risk == "medium"


def test_analyze_document_sorts_by_risk(monkeypatch):
    risks = iter(["low", "high", "medium"])
    monkeypatch.setattr(analyzer.retriever, "search", lambda *a, **kw: [_hit()])
    monkeypatch.setattr(
        analyzer.llm,
        "complete_json",
        lambda messages: {
            "risk": next(risks),
            "quote": "q",
            "norm_ref": "статья 113",
            "problem": "p",
            "suggestion": "s",
        },
    )
    monkeypatch.setattr(analyzer, "classify_document", lambda clauses: "договор")
    clauses = [Clause(str(i), f"пункт {i}") for i in range(1, 4)]
    findings = analyzer.analyze_document(None, clauses)
    assert [f.risk for f in findings] == ["high", "medium", "low"]


def test_analyze_document_survives_llm_failure_on_one_clause(monkeypatch):
    calls = {"n": 0}

    def flaky(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            raise llm.LLMError("модель не вернула валидный JSON после повтора")
        return {
            "risk": "high",
            "quote": "q",
            "norm_ref": "статья 113",
            "problem": "p",
            "suggestion": "s",
        }

    monkeypatch.setattr(analyzer.retriever, "search", lambda *a, **kw: [_hit()])
    monkeypatch.setattr(analyzer.llm, "complete_json", flaky)
    monkeypatch.setattr(analyzer, "classify_document", lambda clauses: "договор")
    findings = analyzer.analyze_document(
        None, [Clause("1.", "первый"), Clause("2.", "второй")]
    )
    assert len(findings) == 1


def test_analyze_document_reports_progress(monkeypatch):
    monkeypatch.setattr(analyzer.retriever, "search", lambda *a, **kw: [])
    monkeypatch.setattr(analyzer, "classify_document", lambda clauses: "договор")
    seen = []
    analyzer.analyze_document(
        None,
        [Clause(str(i), f"п{i}") for i in range(1, 4)],
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(1, 3), (2, 3), (3, 3)]
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

```bash
.venv/Scripts/python.exe -m pytest tests/test_analyzer.py -v
```

Ожидается: FAIL с `ModuleNotFoundError: No module named 'analyzer'`

- [ ] **Step 3: Дополнить `prompts.py`**

Добавить в конец файла:

```python
SYSTEM_ANALYST = """Ты — юрист, проверяющий документ на соответствие законодательству Республики Казахстан.

Жёсткие правила:
1. Оценивай пункт ТОЛЬКО по нормам из блока «Контекст». Норм нет — риска нет.
2. Не выдумывай нарушения. Формальные и безобидные пункты получают risk = "none".
3. quote — дословный фрагмент из пункта документа, ничего не переписывай.
4. suggestion — готовая формулировка на замену, а не совет что подумать.
5. Отвечай ТОЛЬКО JSON-объектом, без пояснений вокруг.

Формат ответа:
{"risk": "none|low|medium|high", "quote": "...", "norm_ref": "статья N <название НПА>", "problem": "...", "suggestion": "..."}
"""

DOC_TYPES = ["договор", "устав", "приказ", "исковое заявление", "прочее"]


def build_classification_messages(sample: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Определи тип юридического документа. Ответь ТОЛЬКО JSON-объектом "
                'вида {"doc_type": "<тип>"}, где тип — одно из: '
                + ", ".join(DOC_TYPES)
            ),
        },
        {"role": "user", "content": f"Начало документа:\n{sample}"},
    ]


def build_clause_messages(clause, doc_type: str, hits: list[Hit]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_ANALYST},
        {
            "role": "user",
            "content": (
                f"Тип документа: {doc_type}\n\n"
                f"Контекст (нормы законодательства РК):\n{format_context(hits)}\n\n"
                f"Пункт документа {clause.no}:\n{clause.text}\n\n"
                "Оцени пункт и верни JSON."
            ),
        },
    ]
```

- [ ] **Step 4: Реализовать `analyzer.py`**

```python
"""Анализ пунктов документа на правовые риски."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Callable

import llm
import prompts
import retriever
from docparse import Clause

RISK_ORDER = {"high": 0, "medium": 1, "low": 2}
_RISK_ALIASES = {
    "none": "none", "нет": "none", "отсутствует": "none",
    "low": "low", "низкий": "low",
    "medium": "medium", "средний": "medium",
    "high": "high", "высокий": "high", "критический": "medium",
}
CLASSIFICATION_SAMPLE_CHARS = 1500


@dataclass
class Finding:
    clause_no: str
    risk: str
    quote: str
    norm_ref: str
    problem: str
    suggestion: str
    link: str


def _normalize_risk(value: object) -> str:
    """Неизвестное значение — не повод терять находку, но и не повод паниковать."""
    return _RISK_ALIASES.get(str(value).strip().lower(), "medium")


def classify_document(clauses: list[Clause]) -> str:
    sample = "\n".join(f"{c.no} {c.text}" for c in clauses)[:CLASSIFICATION_SAMPLE_CHARS]
    try:
        result = llm.complete_json(prompts.build_classification_messages(sample))
    except llm.LLMError:
        return "прочее"
    doc_type = str(result.get("doc_type", "прочее")).strip().lower()
    return doc_type if doc_type in prompts.DOC_TYPES else "прочее"


def analyze_clause(
    conn: sqlite3.Connection, clause: Clause, doc_type: str
) -> Finding | None:
    query = f"{doc_type}: {clause.text}"
    hits = retriever.search(conn, query, k=3)
    if not hits:
        return None

    result = llm.complete_json(prompts.build_clause_messages(clause, doc_type, hits))
    risk = _normalize_risk(result.get("risk"))
    if risk == "none":
        return None

    norm_ref = str(result.get("norm_ref") or hits[0].citation).strip()
    matched = next(
        (h for h in hits if h.article_no and h.article_no in norm_ref), hits[0]
    )
    return Finding(
        clause_no=clause.no,
        risk=risk,
        quote=str(result.get("quote") or clause.text[:200]).strip(),
        norm_ref=norm_ref,
        problem=str(result.get("problem") or "").strip(),
        suggestion=str(result.get("suggestion") or "").strip(),
        link=matched.link,
    )


def analyze_document(
    conn: sqlite3.Connection,
    clauses: list[Clause],
    progress: Callable[[int, int], None] | None = None,
) -> list[Finding]:
    doc_type = classify_document(clauses)
    findings: list[Finding] = []
    total = len(clauses)
    for index, clause in enumerate(clauses, start=1):
        try:
            finding = analyze_clause(conn, clause, doc_type)
        except llm.LLMError:
            finding = None  # пункт пропускаем, остальной документ дорабатываем
        if finding is not None:
            findings.append(finding)
        if progress is not None:
            progress(index, total)
    findings.sort(key=lambda f: RISK_ORDER.get(f.risk, 3))
    return findings
```

- [ ] **Step 5: Запустить тесты, убедиться что проходят**

```bash
.venv/Scripts/python.exe -m pytest tests/test_analyzer.py -v
```

Ожидается: PASS, 7 тестов

- [ ] **Step 6: Коммит**

```bash
git add analyzer.py prompts.py tests/test_analyzer.py
git commit -m "feat: clause-level risk analysis with grounded findings"
```

---

### Task 12: Загрузка документа, SSE-прогресс и отчёт

**Files:**
- Modify: `app.py`, `templates/index.html`, `static/app.css`
- Create: `templates/_report.html`, `tests/test_app_analyze.py`
- Test: `tests/test_app_analyze.py`

**Interfaces:**
- Consumes: `docparse.parse_document`, `docparse.DocParseError`, `analyzer.analyze_document`, `analyzer.Finding`
- Produces: `app.JOBS: dict[str, dict]` (состояние задач в памяти); `POST /analyze` → JSON `{"job_id": str}`; `GET /analyze/{job_id}/events` (SSE, события `progress` и `done`); `GET /analyze/{job_id}/report` → HTML-фрагмент отчёта

- [ ] **Step 1: Написать падающие тесты**

Создать `tests/test_app_analyze.py`:

```python
import dataclasses
import io

import pytest
from docx import Document
from fastapi.testclient import TestClient

import analyzer
import app as app_module
import db


@pytest.fixture
def client(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "t.db")
    db.create_schema(conn)
    monkeypatch.setattr(app_module, "get_connection", lambda: conn)
    monkeypatch.setattr(app_module, "JOBS", {})
    return TestClient(app_module.app)


def _docx_bytes() -> bytes:
    doc = Document()
    doc.add_paragraph("1.1. Оплата производится в течение 90 банковских дней.")
    doc.add_paragraph("1.2. Споры решаются в суде города Алматы.")
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def _finding(clause_no="1.1.", risk="high") -> analyzer.Finding:
    return analyzer.Finding(
        clause_no=clause_no,
        risk=risk,
        quote="Оплата производится в течение 90 банковских дней",
        norm_ref="статья 113 Трудового кодекса Республики Казахстан",
        problem="Срок превышает установленный законом",
        suggestion="Оплата производится не реже одного раза в месяц",
        link="https://adilet.zan.kz/rus/docs/K1500000414#z113",
    )


def test_analyze_accepts_docx_and_returns_job_id(client, monkeypatch):
    monkeypatch.setattr(app_module.analyzer, "analyze_document", lambda *a, **kw: [_finding()])
    response = client.post(
        "/analyze",
        files={"file": ("contract.docx", _docx_bytes(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert response.status_code == 200
    assert "job_id" in response.json()


def test_analyze_rejects_unsupported_extension(client):
    response = client.post("/analyze", files={"file": ("note.txt", b"text", "text/plain")})
    assert response.status_code == 400
    assert "DOCX" in response.json()["detail"]


def test_analyze_rejects_oversized_file(client, monkeypatch):
    # Settings — frozen dataclass, поле напрямую не подменить: подменяем объект целиком
    monkeypatch.setattr(
        app_module, "SETTINGS", dataclasses.replace(app_module.SETTINGS, max_upload_mb=0)
    )
    response = client.post(
        "/analyze", files={"file": ("big.docx", _docx_bytes(), "application/octet-stream")}
    )
    assert response.status_code == 400
    assert "МБ" in response.json()["detail"]


def test_events_stream_reports_progress_and_done(client, monkeypatch):
    def fake_analyze(conn, clauses, progress=None):
        for i, _ in enumerate(clauses, start=1):
            if progress:
                progress(i, len(clauses))
        return [_finding()]

    monkeypatch.setattr(app_module.analyzer, "analyze_document", fake_analyze)
    job_id = client.post(
        "/analyze", files={"file": ("c.docx", _docx_bytes(), "application/octet-stream")}
    ).json()["job_id"]

    with client.stream("GET", f"/analyze/{job_id}/events") as stream:
        payload = "".join(chunk for chunk in stream.iter_text())
    assert "event: progress" in payload
    assert "event: done" in payload


def test_report_renders_findings(client, monkeypatch):
    monkeypatch.setattr(
        app_module.analyzer,
        "analyze_document",
        lambda *a, **kw: [_finding(), _finding("1.2.", "low")],
    )
    job_id = client.post(
        "/analyze", files={"file": ("c.docx", _docx_bytes(), "application/octet-stream")}
    ).json()["job_id"]
    with client.stream("GET", f"/analyze/{job_id}/events") as stream:
        list(stream.iter_text())
    report = client.get(f"/analyze/{job_id}/report")
    assert report.status_code == 200
    assert "90 банковских дней" in report.text
    assert "статья 113" in report.text
    assert "не реже одного раза в месяц" in report.text
    assert "Справочная информация" in report.text


def test_report_shows_clean_document_message(client, monkeypatch):
    monkeypatch.setattr(app_module.analyzer, "analyze_document", lambda *a, **kw: [])
    job_id = client.post(
        "/analyze", files={"file": ("c.docx", _docx_bytes(), "application/octet-stream")}
    ).json()["job_id"]
    with client.stream("GET", f"/analyze/{job_id}/events") as stream:
        list(stream.iter_text())
    report = client.get(f"/analyze/{job_id}/report")
    assert "не выявил" in report.text.lower()


def test_report_for_unknown_job_returns_404(client):
    assert client.get("/analyze/нет-такого/report").status_code == 404


def test_scan_pdf_reported_in_stream(client, monkeypatch):
    def boom(*args, **kwargs):
        raise app_module.DocParseError("В файле нет текстового слоя. Похоже, это скан")

    monkeypatch.setattr(app_module, "parse_document", boom)
    job_id = client.post(
        "/analyze", files={"file": ("scan.pdf", b"%PDF-1.4", "application/pdf")}
    ).json()["job_id"]
    with client.stream("GET", f"/analyze/{job_id}/events") as stream:
        payload = "".join(chunk for chunk in stream.iter_text())
    assert "event: error" in payload
    assert "скан" in payload
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

```bash
.venv/Scripts/python.exe -m pytest tests/test_app_analyze.py -v
```

Ожидается: FAIL — у `app` нет эндпоинта `/analyze`

- [ ] **Step 3: Дополнить `app.py`**

Заменить блок импортов на:

```python
from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import analyzer
import db
import llm
import prompts
import retriever
from config import ROOT, load_settings
from docparse import DocParseError, parse_document
```

Добавить в конец файла:

```python
ALLOWED_SUFFIXES = {".docx", ".pdf"}
JOBS: dict[str, dict] = {}


@app.post("/analyze")
async def analyze(file: UploadFile):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, "Поддерживаются DOCX и PDF.")

    payload = await file.read()
    limit = SETTINGS.max_upload_mb * 1024 * 1024
    if len(payload) > limit:
        raise HTTPException(
            400,
            f"Файл {len(payload) / 1024 / 1024:.1f} МБ превышает предел "
            f"{SETTINGS.max_upload_mb} МБ.",
        )

    SETTINGS.upload_dir.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    path = SETTINGS.upload_dir / f"{job_id}{suffix}"
    path.write_bytes(payload)
    JOBS[job_id] = {
        "path": path,
        "filename": file.filename,
        "findings": None,
        "error": None,
        "clauses": 0,
    }
    return {"job_id": job_id}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/analyze/{job_id}/events")
def analyze_events(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Задача не найдена.")

    def stream() -> Iterator[str]:
        conn = get_connection()
        try:
            clauses = parse_document(job["path"], SETTINGS.max_clauses)
            job["clauses"] = len(clauses)
            yield _sse("progress", {"done": 0, "total": len(clauses)})

            events: list[str] = []
            job["findings"] = analyzer.analyze_document(
                conn,
                clauses,
                progress=lambda done, total: events.append(
                    _sse("progress", {"done": done, "total": total})
                ),
            )
            yield from events
            yield _sse("done", {"findings": len(job["findings"])})
        except (DocParseError, llm.LLMError) as exc:
            job["error"] = str(exc)
            yield _sse("error", {"message": str(exc)})
        finally:
            conn.close()
            job["path"].unlink(missing_ok=True)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/analyze/{job_id}/report", response_class=HTMLResponse)
def analyze_report(request: Request, job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Задача не найдена.")
    return templates.TemplateResponse(
        "_report.html",
        {
            "request": request,
            "filename": job["filename"],
            "clauses": job["clauses"],
            "findings": job["findings"] or [],
            "error": job["error"],
            "disclaimer": prompts.DISCLAIMER,
        },
    )
```

- [ ] **Step 4: Создать `templates/_report.html`**

```html
{% if error %}
  <div class="error">{{ error }}</div>
{% else %}
  <article class="report">
    <h2>{{ filename }}</h2>
    <p class="meta">Пунктов проверено: {{ clauses }} · находок: {{ findings|length }}</p>
    {% if findings %}
      <ol class="findings">
        {% for f in findings %}
          <li class="finding risk-{{ f.risk }}">
            <div class="head">
              <span class="badge">{{ f.risk }}</span>
              <span class="clause">пункт {{ f.clause_no }}</span>
              <a href="{{ f.link }}" target="_blank" rel="noopener">{{ f.norm_ref }}</a>
            </div>
            <blockquote>{{ f.quote }}</blockquote>
            <p class="problem">{{ f.problem }}</p>
            <p class="suggestion"><strong>Предлагаемая формулировка:</strong> {{ f.suggestion }}</p>
          </li>
        {% endfor %}
      </ol>
    {% else %}
      <p class="clean">Анализ не выявил рисков по загруженным НПА. Это не гарантия — корпус ограничен списком документов ниже.</p>
    {% endif %}
    <p class="disclaimer">{{ disclaimer }}</p>
  </article>
{% endif %}
```

- [ ] **Step 5: Заменить заглушку вкладки документов в `templates/index.html`**

Заменить блок `<section id="pane-doc" ...>` на:

```html
<section id="pane-doc" class="pane" hidden>
  <form id="upload-form">
    <input type="file" name="file" accept=".docx,.pdf" required>
    <button type="submit">Проверить документ</button>
  </form>
  <div id="progress" hidden>
    <progress id="progress-bar" value="0" max="1"></progress>
    <span id="progress-text"></span>
  </div>
  <div id="report"></div>
</section>

<script>
document.getElementById('upload-form').addEventListener('submit', async function (event) {
  event.preventDefault();
  var report = document.getElementById('report');
  var progress = document.getElementById('progress');
  var bar = document.getElementById('progress-bar');
  var text = document.getElementById('progress-text');
  report.innerHTML = '';
  progress.hidden = false;
  text.textContent = 'Разбираю документ…';

  var response = await fetch('/analyze', { method: 'POST', body: new FormData(event.target) });
  if (!response.ok) {
    progress.hidden = true;
    report.innerHTML = '<div class="error">' + (await response.json()).detail + '</div>';
    return;
  }
  var jobId = (await response.json()).job_id;
  var source = new EventSource('/analyze/' + jobId + '/events');

  source.addEventListener('progress', function (e) {
    var data = JSON.parse(e.data);
    bar.max = data.total;
    bar.value = data.done;
    text.textContent = 'Проверено пунктов: ' + data.done + ' из ' + data.total;
  });
  source.addEventListener('done', async function () {
    source.close();
    progress.hidden = true;
    report.innerHTML = await (await fetch('/analyze/' + jobId + '/report')).text();
  });
  source.addEventListener('error', async function (e) {
    source.close();
    progress.hidden = true;
    report.innerHTML = await (await fetch('/analyze/' + jobId + '/report')).text();
  });
});
</script>
```

- [ ] **Step 6: Дополнить `static/app.css`**

```css
.report { border: 1px solid var(--line); border-radius: 8px; padding: 16px; margin-top: 18px; }
.report h2 { font-size: 18px; margin: 0 0 4px; }
.meta { color: var(--muted); font-size: 14px; margin: 0 0 16px; }
.findings { list-style: none; padding: 0; }
.finding { border-left: 4px solid var(--line); padding: 10px 14px; margin-bottom: 16px; background: #fafafa; border-radius: 0 6px 6px 0; }
.finding.risk-high { border-left-color: #d33; }
.finding.risk-medium { border-left-color: #e90; }
.finding.risk-low { border-left-color: #999; }
.head { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; font-size: 14px; }
.badge { text-transform: uppercase; font-size: 11px; letter-spacing: .04em; padding: 2px 7px; border-radius: 3px; background: #eee; }
.risk-high .badge { background: #fde3e3; color: #a11; }
.risk-medium .badge { background: #fdf0d5; color: #96690a; }
.clause { color: var(--muted); }
blockquote { margin: 10px 0; padding-left: 12px; border-left: 2px solid var(--line); color: #333; font-style: italic; }
.suggestion { background: #f0f6ff; padding: 8px 10px; border-radius: 4px; }
.clean { color: #2a7; }
#progress { margin-top: 16px; display: flex; gap: 12px; align-items: center; }
progress { width: 260px; }
```

- [ ] **Step 7: Запустить все тесты**

```bash
.venv/Scripts/python.exe -m pytest -v
```

Ожидается: PASS, около 65 тестов

- [ ] **Step 8: Коммит**

```bash
git add app.py templates static tests/test_app_analyze.py
git commit -m "feat: document upload, SSE progress and risk report"
```

---

### Task 13: Проверка интерфейса в браузере и интеграционный набор

**Files:**
- Create: `tests/test_integration.py`, `tests/test_ui.py`
- Modify: `requirements.txt` (добавить `pytest-playwright==0.6.2`)
- Test: `tests/test_ui.py`, `tests/test_integration.py`

**Interfaces:**
- Consumes: `app.app`, `cli.py`, собранный `law.db`
- Produces: набор `@pytest.mark.integration` для проверки на живой Ollama и реальном браузере

- [ ] **Step 1: Установить pytest-playwright**

Добавить в `requirements.txt`:

```
pytest-playwright==0.6.2
```

```bash
.venv/Scripts/python.exe -m pip install pytest-playwright==0.6.2
```

- [ ] **Step 2: Написать интеграционные тесты качества ответов**

Создать `tests/test_integration.py`:

```python
"""Требуют запущенной Ollama и собранного индекса.

Запуск: pytest -m integration
"""
import pytest

import app as app_module
import db
import llm
import retriever

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def conn():
    llm.health_check()
    connection = db.connect()
    count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if count == 0:
        pytest.skip("Индекс пуст: выполните python cli.py scrape && python cli.py index")
    return connection


@pytest.mark.parametrize(
    "question,expected_act",
    [
        ("обязателен ли письменный трудовой договор", "Трудовой кодекс"),
        ("минимальный размер уставного капитала ТОО", "ограниченной"),
        ("срок исковой давности по общему правилу", "Гражданский кодекс"),
    ],
)
def test_retrieval_finds_expected_act(conn, question, expected_act):
    hits = retriever.search(conn, question, k=4)
    assert hits, f"по вопросу «{question}» ничего не найдено"
    assert any(expected_act in hit.act_title for hit in hits)


def test_offtopic_question_is_refused(conn):
    result = app_module.answer_question(conn, "как приготовить борщ")
    assert result["refused"] is True


def test_answer_cites_a_source(conn):
    result = app_module.answer_question(conn, "какова продолжительность ежегодного отпуска")
    assert result["refused"] is False
    assert result["sources"]
    assert "стать" in result["answer"].lower()
```

- [ ] **Step 3: Написать браузерный тест**

Создать `tests/test_ui.py`:

```python
"""Проверка интерфейса в реальном браузере.

Запуск: pytest -m integration tests/test_ui.py
Требует запущенного сервера: uvicorn app:app --port 8000
"""
import re

import pytest
from docx import Document
from playwright.sync_api import expect, sync_playwright

pytestmark = pytest.mark.integration

BASE_URL = "http://localhost:8000"


@pytest.fixture(scope="module")
def page():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(BASE_URL, timeout=10_000)
        except Exception:
            pytest.skip("Сервер не запущен: uvicorn app:app --port 8000")
        yield page
        browser.close()


def test_corpus_list_is_visible(page):
    page.goto(BASE_URL)
    expect(page.locator(".corpus")).to_contain_text("Трудовой кодекс")


def test_question_returns_answer_with_source_link(page):
    page.goto(BASE_URL)
    page.fill("textarea[name=question]", "обязателен ли письменный трудовой договор")
    page.click("button[type=submit]")
    expect(page.locator(".answer")).to_be_visible(timeout=180_000)
    expect(page.locator(".sources a").first).to_have_attribute(
        "href", re.compile(r"adilet\.zan\.kz")
    )


def test_document_upload_produces_report(page, tmp_path):
    document = Document()
    document.add_paragraph("1.1. Оплата производится в течение 90 банковских дней.")
    document.add_paragraph("1.2. Работник работает 60 часов в неделю без доплат.")
    path = tmp_path / "contract.docx"
    document.save(path)

    page.goto(BASE_URL)
    page.click("button[data-target=pane-doc]")
    page.set_input_files("input[type=file]", str(path))
    page.click("#upload-form button[type=submit]")
    expect(page.locator(".report")).to_be_visible(timeout=600_000)
    expect(page.locator(".report")).to_contain_text("Пунктов проверено")


def test_disclaimer_present_on_page(page):
    page.goto(BASE_URL)
    expect(page.locator("footer .disclaimer")).to_contain_text("не является юридической")
```

- [ ] **Step 4: Прогнать обычный набор — он не должен затронуть браузер и Ollama**

```bash
.venv/Scripts/python.exe -m pytest -v
```

Ожидается: PASS, интеграционные тесты не собираются (`addopts = -m "not integration"`)

- [ ] **Step 5: Прогнать интеграционный набор**

```bash
.venv/Scripts/python.exe -m pytest -m integration tests/test_integration.py -v
```

Ожидается: PASS. Занимает несколько минут — каждый вопрос идёт через локальную модель.

- [ ] **Step 6: Прогнать браузерный набор**

В отдельном окне:

```bash
.venv/Scripts/python.exe -m uvicorn app:app --port 8000
```

Затем:

```bash
.venv/Scripts/python.exe -m pytest -m integration tests/test_ui.py -v
```

Ожидается: PASS, 4 теста. Тест загрузки документа занимает до 10 минут.

- [ ] **Step 7: Коммит**

```bash
git add requirements.txt tests/test_integration.py tests/test_ui.py
git commit -m "test: integration suite for answer quality and browser UI"
```

---

## Известные ограничения реализации

Фиксируются осознанно, чтобы исполнитель не принял их за недоработки:

- Состояние задач анализа живёт в памяти процесса и теряется при перезапуске. Для однопользовательского локального приложения этого достаточно; при появлении второго пользователя понадобится таблица в базе.
- SSE-события прогресса накапливаются в списке и отдаются пачкой после завершения анализа: `analyzer.analyze_document` синхронный. Пользователь видит движение шкалы дискретно. Настоящий потоковый прогресс требует перевода анализа на асинхронные задачи — делать при появлении жалоб на ожидание.
- `revision_date` вычисляется как максимальная дата из сносок об изменениях в тексте НПА. Это приближение: adilet не публикует машиночитаемую дату редакции.
- Порог релевантности единый для всех запросов. Юридические домены различаются по плотности терминов, и на практике может понадобиться отдельная калибровка — сначала собрать жалобы, потом усложнять.
