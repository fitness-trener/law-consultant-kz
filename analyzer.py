"""Анализ пунктов документа на правовые риски."""
from __future__ import annotations

import re
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
# Номер статьи может нести суффикс, как в "152-1".
_ARTICLE_NO_RE = re.compile(r"статья\s+(\d+(?:-\d+)?)", re.IGNORECASE)


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
    if value is None:
        # Модель не вернула вердикт вообще — намеренно считаем это отсутствием
        # риска, а не поводом эскалировать: придуманный риск хуже упущенного,
        # значит непонятное отсутствие ответа тоже должно склоняться к "нет риска".
        return "none"
    return _RISK_ALIASES.get(str(value).strip().lower(), "medium")


def _article_no_from_ref(norm_ref: str) -> str | None:
    """Номер статьи из ссылки модели, если она вообще его называет."""
    match = _ARTICLE_NO_RE.search(norm_ref)
    return match.group(1) if match else None


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
    article_no = _article_no_from_ref(norm_ref)
    candidates = [h for h in hits if h.article_no == article_no] if article_no else []
    # Несколько чанков одной статьи среди hits — берём лучший по скору. Если
    # модель не назвала статью или назвала ту, которой нет среди hits, значит
    # именно по верхнему hit пункт и оценивался — это и есть fallback.
    matched = max(candidates, key=lambda h: h.score) if candidates else hits[0]
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
