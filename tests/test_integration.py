"""Требуют запущенной Ollama и собранного индекса.

Запуск: pytest -m integration tests/test_integration.py
"""
import sqlite3

import pytest

import app as app_module
import db
import llm
import retriever

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def conn():
    try:
        llm.health_check()
    except llm.LLMError as exc:
        pytest.skip(f"Ollama недоступна: {exc}")

    connection = db.connect()
    try:
        count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    except sqlite3.OperationalError as exc:
        pytest.skip(f"Индекс не собран: {exc}")
    if count == 0:
        pytest.skip("Индекс пуст: выполните python cli.py scrape && python cli.py index")
    return connection


@pytest.mark.parametrize(
    "question,expected_act",
    [
        ("обязателен ли письменный трудовой договор", "Трудовой кодекс"),
        # Развёрнутая формулировка, не аббревиатура: «ТОО» в тексте закона не
        # встречается — он всюду пишет «товарищество с ограниченной
        # ответственностью», а FTS5 работает без стемминга и синонимов, так что
        # по аббревиатуре вытягивает только вектор, и тот уводит к закону об АО.
        ("уставный капитал товарищества с ограниченной ответственностью", "ограниченной"),
        ("срок исковой давности по общему правилу", "Гражданский кодекс"),
    ],
)
def test_retrieval_finds_expected_act(conn, question, expected_act):
    hits = retriever.search(conn, question, k=4)
    assert hits, f"по вопросу «{question}» ничего не найдено"
    assert any(expected_act in hit.act_title for hit in hits)


def test_offtopic_question_is_refused(conn):
    # ponytail: "рецепт борща республики казахстан" — заведомо отказной запрос,
    # ранее подтверждённый вручную на реальном корпусе. Не заменять на другие
    # бытовые формулировки без повторной проверки: отказ зависит от того,
    # перебивает ли хоть один термин запроса порог DISCRIMINATIVE_DF_FRACTION
    # в retriever._is_discriminative, а это свойство корпуса, не намерения.
    result = app_module.answer_question(conn, "рецепт борща республики казахстан")
    assert result["refused"] is True


def test_answer_cites_a_source(conn):
    result = app_module.answer_question(conn, "какова продолжительность ежегодного отпуска")
    assert result["refused"] is False
    assert result["sources"]
    assert "стать" in result["answer"].lower()
