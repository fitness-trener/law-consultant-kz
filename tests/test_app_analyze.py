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
    # docparse.parse_document treats extracted text under 200 chars as a scan
    # without a text layer (see docparse.MIN_TEXT_LENGTH) — these tests don't
    # mock parse_document, so the fixture text must clear that bar for real.
    doc = Document()
    doc.add_paragraph(
        "1.1. Оплата производится в течение 90 банковских дней с момента "
        "подписания настоящего договора обеими сторонами и предоставления "
        "соответствующих закрывающих документов в полном объёме."
    )
    doc.add_paragraph(
        "1.2. Споры и разногласия, возникающие при исполнении настоящего "
        "договора, решаются в суде города Алматы в соответствии с "
        "законодательством Республики Казахстан."
    )
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
