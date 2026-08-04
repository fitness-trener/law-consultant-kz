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
