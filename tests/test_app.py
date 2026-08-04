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
    assert prompts.DISCLAIMER[:30] in response.text


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
    assert prompts.DISCLAIMER[:30] in response.text
    # Pin: a generic exception handler must not flatten this actionable,
    # cause-specific message into the generic "unexpected error" text.
    assert app_module.GENERIC_ERROR not in response.text


def test_chat_endpoint_handles_unexpected_error(tmp_path, monkeypatch):
    """retriever.search runs an embedding call and two SQL queries — at least
    as likely to fail as llm.complete, which already gets friendly handling.
    A bare RuntimeError here must not leak to a Starlette default 500 page."""
    monkeypatch.setattr(app_module, "get_connection", lambda: _seed(tmp_path))

    def boom(*args, **kwargs):
        raise RuntimeError("índex corrupted: disk I/O error")

    monkeypatch.setattr(app_module.retriever, "search", boom)
    client = TestClient(app_module.app, raise_server_exceptions=False)
    response = client.post("/chat", data={"question": "вопрос"})
    assert response.status_code == 200
    assert "disk I/O error" not in response.text
    assert "Internal Server Error" not in response.text
    assert prompts.DISCLAIMER[:30] in response.text


def test_index_page_lists_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "get_connection", lambda: _seed(tmp_path))
    client = TestClient(app_module.app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Трудовой кодекс" in response.text


def test_index_page_handles_db_error(tmp_path, monkeypatch):
    """db.corpus_summary raising must not leak Starlette's bare 500 page —
    the user must still see a Russian message with the disclaimer."""
    monkeypatch.setattr(app_module, "get_connection", lambda: _seed(tmp_path))

    def boom(conn):
        raise RuntimeError("database disk image is malformed")

    monkeypatch.setattr(app_module.db, "corpus_summary", boom)
    client = TestClient(app_module.app, raise_server_exceptions=False)
    response = client.get("/")
    assert response.status_code == 200
    assert "database disk image is malformed" not in response.text
    assert "Internal Server Error" not in response.text
    assert prompts.DISCLAIMER[:30] in response.text


class _ClosableConnProxy:
    """sqlite3.Connection instances don't allow per-instance attribute
    overrides (its methods are read-only slots), so wrap it to observe
    whether .close() was actually called."""

    def __init__(self, real):
        self._real = real
        self.closed = False

    def close(self):
        self.closed = True
        self._real.close()

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_index_closes_connection_even_when_it_raises(tmp_path, monkeypatch):
    """The connection must be closed even when the route raises — assert
    this directly instead of inferring it from the response."""
    proxy = _ClosableConnProxy(_seed(tmp_path))
    monkeypatch.setattr(app_module, "get_connection", lambda: proxy)

    def boom(_conn):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module.db, "corpus_summary", boom)
    client = TestClient(app_module.app, raise_server_exceptions=False)
    client.get("/")
    assert proxy.closed is True
