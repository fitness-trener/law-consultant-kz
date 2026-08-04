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


def test_preamble_before_first_clause_is_kept(contract):
    clauses = docparse.parse_document(contract)
    assert any(
        c.no == docparse.PREAMBLE_CLAUSE_NO
        and "ДОГОВОР ВОЗМЕЗДНОГО ОКАЗАНИЯ УСЛУГ" in c.text
        for c in clauses
    )
    # numbered clauses still parse correctly around the preamble
    assert [c.no for c in clauses if c.no != docparse.PREAMBLE_CLAUSE_NO] == [
        "1.",
        "1.1.",
        "1.2.",
        "2.",
        "2.1.",
        "2.2.",
    ]


@pytest.fixture
def contract_with_table(tmp_path) -> Path:
    doc = Document()
    doc.add_paragraph("ДОГОВОР ВОЗМЕЗДНОГО ОКАЗАНИЯ УСЛУГ")
    doc.add_paragraph("1. Предмет договора")
    doc.add_paragraph(
        "1.1. Исполнитель обязуется оказать услуги по разработке "
        "программного обеспечения в соответствии с техническим заданием."
    )
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Услуга"
    table.cell(0, 1).text = "Цена"
    doc.add_paragraph("2. Стоимость и порядок расчётов")
    doc.add_paragraph(
        "2.1. Стоимость услуг составляет 1 000 000 тенге и оплачивается "
        "в течение 90 банковских дней с момента подписания акта."
    )
    path = tmp_path / "contract_with_table.docx"
    doc.save(path)
    return path


def test_docx_table_stays_with_preceding_clause(contract_with_table):
    clauses = docparse.parse_document(contract_with_table)
    by_no = {c.no: c.text for c in clauses}
    assert "Услуга | Цена" in by_no["1.1."]
    assert "Услуга" not in by_no["2.1."]
    assert "Цена" not in by_no["2.1."]


def test_pdf_with_sliver_of_text_reports_scan(tmp_path, monkeypatch):
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    # Stray OCR noise / page number: real but far too short to be a contract.
    monkeypatch.setattr(docparse, "_pdf_text", lambda p: "Стр. 3")
    with pytest.raises(docparse.DocParseError, match="скан"):
        docparse.parse_document(path)


def test_no_text_loss(contract):
    text = docparse.extract_text(contract)
    clauses = docparse.parse_document(contract)
    produced = " ".join(f"{c.no} {c.text}" for c in clauses)
    for word in text.split():
        assert word in produced
