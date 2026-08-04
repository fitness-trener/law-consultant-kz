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


def test_health_check_passes_with_default_tag(monkeypatch):
    """Ollama отдаёт модели без явного тега как name:latest — bge-m3 в конфиге
    должен совпасть с bge-m3:latest в /api/tags."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "bge-m3:latest"},
                    {"name": "qwen3:4b-instruct"},
                ]
            },
        )

    monkeypatch.setattr(llm, "_transport", _mock_transport(handler))
    llm.health_check()


def test_health_check_raises_for_missing_model(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "qwen3:4b-instruct"}]})

    monkeypatch.setattr(llm, "_transport", _mock_transport(handler))
    with pytest.raises(llm.LLMError, match="bge-m3"):
        llm.health_check()
