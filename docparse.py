"""Разбор загруженных документов на пункты."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

CLAUSE_RE = re.compile(r"(?m)^\s*((?:\d+\.)+|\d+\)|Статья\s+\d+\.|§\s*\d+\.)\s*")

# Clause number given to text preceding the first numbered clause (title,
# parties, preamble). Distinct from any real numbering the regex above matches.
PREAMBLE_CLAUSE_NO = "преамбула"

# Below this length, extracted PDF text is treated as scan noise (stray page
# numbers, header fragments, OCR artefacts) rather than a real text layer.
MIN_TEXT_LENGTH = 200


class DocParseError(RuntimeError):
    """Документ невозможно разобрать — текст показывается пользователю."""


@dataclass
class Clause:
    no: str
    text: str


def _docx_text(path: Path) -> str:
    from docx import Document
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(str(path))
    lines: list[str] = []
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            text = Paragraph(child, document).text.strip()
            if text:
                lines.append(text)
        elif child.tag == qn("w:tbl"):
            table = Table(child, document)
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    lines.append(" | ".join(cells))
    return "\n".join(lines)


def _pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _docx_text(path)
    if suffix == ".pdf":
        return _pdf_text(path)
    raise DocParseError(
        f"Формат {suffix or 'без расширения'} не поддерживается. "
        "Поддерживаются DOCX и PDF."
    )


def split_clauses(text: str, max_clauses: int = 300) -> list[Clause]:
    matches = list(CLAUSE_RE.finditer(text))
    clauses: list[Clause] = []

    if len(matches) >= 2:
        leading = re.sub(r"\s+", " ", text[: matches[0].start()]).strip()
        if leading:
            clauses.append(Clause(no=PREAMBLE_CLAUSE_NO, text=leading))
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[match.end() : end].strip()
            body = re.sub(r"\s+", " ", body)
            if body:
                clauses.append(Clause(no=match.group(1).strip(), text=body))
    else:
        for i, line in enumerate(
            [ln.strip() for ln in text.split("\n") if ln.strip()], start=1
        ):
            clauses.append(Clause(no=str(i), text=line))

    if len(clauses) > max_clauses:
        raise DocParseError(
            f"Слишком много пунктов: {len(clauses)}, предел {max_clauses}. "
            "Загрузите документ по частям."
        )
    return clauses


def parse_document(path: Path, max_clauses: int = 300) -> list[Clause]:
    text = extract_text(path)
    if len(text.strip()) < MIN_TEXT_LENGTH:
        raise DocParseError(
            "В файле нет текстового слоя. Похоже, это скан — "
            "распознавание изображений не поддерживается."
        )
    clauses = split_clauses(text, max_clauses)
    if not clauses:
        raise DocParseError("Не удалось выделить ни одного пункта документа.")
    return clauses
