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
        pg = browser.new_page()
        try:
            pg.goto(BASE_URL, timeout=10_000)
        except Exception:
            pytest.skip("Сервер не запущен: uvicorn app:app --port 8000")
        yield pg
        browser.close()


def test_corpus_list_is_visible(page):
    page.goto(BASE_URL)
    expect(page.locator(".corpus")).to_contain_text("Трудовой кодекс")


def test_question_returns_answer_with_source_link(page):
    page.goto(BASE_URL)
    page.fill("textarea[name=question]", "обязателен ли письменный трудовой договор")
    # ponytail: scoped to #pane-chat — the doc-upload pane has its own
    # button[type=submit], so the bare selector is ambiguous (strict-mode
    # violation) even though that pane is hidden at the time of the click.
    page.click("#pane-chat button[type=submit]")
    expect(page.locator(".answer")).to_be_visible(timeout=180_000)
    expect(page.locator(".sources a").first).to_have_attribute(
        "href", re.compile(r"adilet\.zan\.kz")
    )


def test_document_upload_produces_report(page, tmp_path):
    # Real-looking contract, comfortably over docparse.MIN_TEXT_LENGTH (200
    # chars), with clauses that actually violate the Трудовой кодекс so the
    # analysis produces a finding instead of an empty report.
    document = Document()
    document.add_paragraph("ДОГОВОР ОКАЗАНИЯ УСЛУГ № 14")
    document.add_paragraph(
        "Настоящий договор заключён между Заказчиком и Исполнителем "
        "о нижеследующем."
    )
    document.add_paragraph(
        "1.1. Оплата услуг производится Заказчиком в течение 90 "
        "банковских дней с момента подписания акта приёма-передачи."
    )
    document.add_paragraph(
        "1.2. Работник обязан работать 60 часов в неделю без доплаты "
        "за сверхурочную работу и без предоставления выходных дней."
    )
    document.add_paragraph(
        "1.3. Работник отказывается от права на ежегодный оплачиваемый "
        "трудовой отпуск на весь срок действия настоящего договора."
    )
    path = tmp_path / "contract.docx"
    document.save(path)

    page.goto(BASE_URL)
    page.click("button[data-target=pane-doc]")
    page.set_input_files("input[type=file]", str(path))
    page.click("#upload-form button[type=submit]")
    # Один вызов модели на пункт (~1 минуту на локальной 4B-модели) — таймаут
    # намеренно большой, документ намеренно короткий (несколько пунктов).
    expect(page.locator(".report")).to_be_visible(timeout=600_000)
    expect(page.locator(".report")).to_contain_text("Пунктов проверено")


def test_document_upload_too_short_shows_scan_message(page, tmp_path):
    # Under docparse.MIN_TEXT_LENGTH (200 chars) — treated as a scanned
    # document with no text layer. Real user path: someone uploads a photo
    # of a contract instead of a text document.
    document = Document()
    document.add_paragraph("Слишком короткий документ.")
    path = tmp_path / "scan.docx"
    document.save(path)

    page.goto(BASE_URL)
    page.click("button[data-target=pane-doc]")
    page.set_input_files("input[type=file]", str(path))
    page.click("#upload-form button[type=submit]")
    expect(page.locator(".report")).to_be_visible(timeout=30_000)
    expect(page.locator(".report")).to_contain_text("скан")


def test_disclaimer_present_on_page(page):
    page.goto(BASE_URL)
    # Дисклеймер начинается с заглавной буквы ("Не является..."),
    # to_contain_text регистрозависим — см. prompts.DISCLAIMER.
    expect(page.locator("footer .disclaimer")).to_contain_text("Не является юридической")
