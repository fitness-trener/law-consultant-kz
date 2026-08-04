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
