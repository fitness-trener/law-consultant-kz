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
# Номер статьи может нести суффикс, как в "152-1". Модель почти никогда не
# пишет именительный "статья" в живом тексте — только косвенные падежи
# ("согласно статье", "на основании статьи", "в соответствии со статьёй",
# "статьями X и Y") и аббревиатуру "ст." Все словоформы "статья" делят общую
# основу "стать" (статья/статье/статьи/статьей/статьёй/статьями/статью), так
# что одна основа + аббревиатура покрывают их без отдельной ветки на каждый
# падеж. \b перед основой не даёт "ст." подхватить хвост чужого слова
# (в "уст. 5" перед "ст" нет границы слова — не совпадёт).
_ARTICLE_WORD_RE = r"(?:стать[а-яё]*|ст\.)"
# Второе число — только для "статьями 88 и 90": число сразу после первого,
# отделённое "и" или запятой. Порядок "88 статьи" в ответах модели не
# встречается, поэтому обратный порядок намеренно не добавлен.
_ARTICLE_NO_RE = re.compile(
    rf"\b{_ARTICLE_WORD_RE}\s*(\d+(?:-\d+)?)(?:\s*(?:,|и)\s*(\d+(?:-\d+)?))?",
    re.IGNORECASE,
)

# Слова, которые встречаются почти в любом названии НПА ("кодекс", "закон",
# "Республики Казахстан") и потому сами по себе не отличают один акт от
# другого — не годятся как признак того, что ссылка модели называет именно
# этот акт.
_ACT_TITLE_STOPWORDS = {
    "кодекс", "кодекса", "кодексу", "кодексе", "кодексом",
    "республики", "республика", "казахстан", "казахстана",
    "закон", "закона", "законе", "законом", "законодательства",
    "рк",
}
_ACT_WORD_RE = re.compile(r"[a-zA-Zа-яёА-ЯЁ]+")
# Длина префикса, по которому сверяем однокоренные словоформы ("Трудового" /
# "Трудовой") без полноценного морфологического разбора.
_ACT_STEM_LEN = 6


@dataclass
class Finding:
    clause_no: str
    risk: str
    quote: str
    norm_ref: str
    problem: str
    suggestion: str
    link: str


@dataclass
class AnalysisResult:
    """Итог анализа документа: находки и то, сколько пунктов не удалось
    по-настоящему оценить (пропали не потому, что рисков нет, а потому, что
    модель не дала внятного ответа или сорвалась с ошибкой)."""

    findings: list[Finding]
    skipped: int


def _normalize_risk(value: object) -> str:
    """Неизвестное значение — не повод терять находку, но и не повод паниковать."""
    if value is None:
        # Модель не вернула вердикт вообще — намеренно считаем это отсутствием
        # риска, а не поводом эскалировать: придуманный риск хуже упущенного,
        # значит непонятное отсутствие ответа тоже должно склоняться к "нет риска".
        return "none"
    return _RISK_ALIASES.get(str(value).strip().lower(), "medium")


def extract_article_numbers(text: str) -> list[str]:
    """Каждый номер статьи, упомянутый в тексте. Общая регулярка для
    analyzer (сверка ссылки с найденными нормами) и app.answer_question
    (сверка ответа консультанта с найденными нормами) — фабрикация номера
    статьи должна ловиться одним и тем же способом в обоих местах."""
    numbers: list[str] = []
    for match in _ARTICLE_NO_RE.finditer(text):
        numbers.extend(group for group in match.groups() if group)
    return numbers


def _article_no_from_ref(norm_ref: str) -> str | None:
    """Номер статьи из ссылки модели, если она вообще его называет."""
    numbers = extract_article_numbers(norm_ref)
    return numbers[0] if numbers else None


def _distinctive_act_words(act_title: str) -> list[str]:
    """Слова названия акта, которые действительно его отличают от других
    (не "кодекс", не "Республики Казахстан")."""
    words = _ACT_WORD_RE.findall(act_title)
    return [w for w in words if len(w) > 3 and w.lower() not in _ACT_TITLE_STOPWORDS]


def _act_stem(word: str) -> str:
    return word.lower()[:_ACT_STEM_LEN]


def _act_agrees(hit: "retriever.Hit", norm_ref: str, hits: list) -> bool:
    """Модель не обязана называть акт явно, но если называет — должен
    называться акт именно этого hit, а не другой акт из числа найденных.
    Статья 113 встречается в 21 кодексе — совпадения одного номера мало,
    чтобы решить, к какому акту относится ссылка."""
    ref_stems = {_act_stem(w) for w in _ACT_WORD_RE.findall(norm_ref)}
    this_words = _distinctive_act_words(hit.act_title)
    if any(_act_stem(w) in ref_stems for w in this_words):
        return True
    for other in hits:
        if other.act_title == hit.act_title:
            continue
        other_words = _distinctive_act_words(other.act_title)
        if any(_act_stem(w) in ref_stems for w in other_words):
            return False  # модель явно назвала другой найденный акт
    return True  # модель вообще не назвала ни один из найденных актов


def classify_document(clauses: list[Clause]) -> str:
    sample = "\n".join(f"{c.no} {c.text}" for c in clauses)[:CLASSIFICATION_SAMPLE_CHARS]
    try:
        result = llm.complete_json(prompts.build_classification_messages(sample))
    except llm.LLMError:
        return "прочее"
    doc_type = str(result.get("doc_type", "прочее")).strip().lower()
    return doc_type if doc_type in prompts.DOC_TYPES else "прочее"


def _evaluate_clause(
    conn: sqlite3.Connection, clause: Clause, doc_type: str
) -> tuple[Finding | None, bool]:
    """Возвращает (находка либо None, пропущен ли пункт). "Пропущен" означает
    "модель не дала внятного вердикта", а не "модель сказала, что риска нет" —
    отчёт должен уметь отличить одно от другого (иначе несостоявшаяся оценка
    выглядит как чистый документ)."""
    query = f"{doc_type}: {clause.text}"
    hits = retriever.search(conn, query, k=3)
    if not hits:
        return None, False

    result = llm.complete_json(prompts.build_clause_messages(clause, doc_type, hits))
    raw_risk = result.get("risk")
    risk = _normalize_risk(raw_risk)
    if risk == "none":
        # raw_risk is None → ключа не было вовсе (или он null) — модель не
        # ответила по существу, это пропуск. Явное "none"/"нет"/"отсутствует"
        # от модели — настоящий вердикт, не пропуск.
        return None, raw_risk is None

    norm_ref = str(result.get("norm_ref") or hits[0].citation).strip()
    article_no = _article_no_from_ref(norm_ref)
    candidates = (
        [h for h in hits if h.article_no == article_no and _act_agrees(h, norm_ref, hits)]
        if article_no
        else []
    )
    # Несколько чанков одной статьи одного акта среди hits — берём лучший по
    # скору. Если модель не назвала статью, назвала статью не из числа hits,
    # или назвала статью с тем же номером, но из другого акта — значит именно
    # по верхнему hit пункт и оценивался, это и есть fallback.
    matched = max(candidates, key=lambda h: h.score) if candidates else hits[0]
    finding = Finding(
        clause_no=clause.no,
        risk=risk,
        quote=str(result.get("quote") or clause.text[:200]).strip(),
        norm_ref=norm_ref,
        problem=str(result.get("problem") or "").strip(),
        suggestion=str(result.get("suggestion") or "").strip(),
        link=matched.link,
    )
    return finding, False


def analyze_clause(
    conn: sqlite3.Connection, clause: Clause, doc_type: str
) -> Finding | None:
    finding, _skipped = _evaluate_clause(conn, clause, doc_type)
    return finding


def analyze_document(
    conn: sqlite3.Connection,
    clauses: list[Clause],
    progress: Callable[[int, int], None] | None = None,
) -> AnalysisResult:
    doc_type = classify_document(clauses)
    findings: list[Finding] = []
    skipped = 0
    total = len(clauses)
    for index, clause in enumerate(clauses, start=1):
        try:
            finding, was_skipped = _evaluate_clause(conn, clause, doc_type)
        except llm.LLMError:
            finding, was_skipped = None, True  # пункт пропускаем, остальной документ дорабатываем
        if finding is not None:
            findings.append(finding)
        if was_skipped:
            skipped += 1
        if progress is not None:
            progress(index, total)
    findings.sort(key=lambda f: RISK_ORDER.get(f.risk, 3))
    return AnalysisResult(findings=findings, skipped=skipped)
