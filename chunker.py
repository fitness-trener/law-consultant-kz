"""Нарезка статей НПА на чанки.

Чанк — статья целиком. Юридический текст уже структурирован, статья
самодостаточна по смыслу, а ссылка на неё берётся из метаданных,
а не угадывается моделью. Длинные статьи режутся по пунктам.
"""
from __future__ import annotations

import re

from models import Act, Article, Chunk

_POINT_RE = re.compile(r"(?m)^\s*(\d+(?:-\d+)?)\.\s+")
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _split_by_points(text: str) -> list[tuple[str | None, str]]:
    """Режет по нумерации пунктов; при её отсутствии — по абзацам.

    Возвращает пары (номер_пункта, текст). Номер — None для абзацного
    и одиночного фолбэков, чтобы не выдумывать нумерацию там, где её нет
    в самом акте (нумерация НПА бывает с пропусками из-за исключённых пунктов).
    """
    matches = list(_POINT_RE.finditer(text))
    if len(matches) >= 2:
        parts = []
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            parts.append((match.group(1), text[match.start() : end]))
        return parts
    paragraphs = [p for p in text.split("\n") if p.strip()]
    if len(paragraphs) >= 2:
        return [(None, p) for p in paragraphs]
    return [(None, text)]


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Последний рубеж: режет по границам предложений в пределах лимита."""
    sentences = re.split(r"(?<=[.;])\s+", text)
    out: list[str] = []
    buf = ""
    for sentence in sentences:
        if buf and len(buf) + len(sentence) + 1 > max_chars:
            out.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if buf:
        out.append(buf)
    return out or [text]


def chunk_article(
    article: Article, max_chars: int = 2000, min_chars: int = 100
) -> list[Chunk]:
    header = f"Статья {article.number}. {article.title}".strip()
    body = _normalize(article.text)

    if len(body) + len(header) <= max_chars:
        return [
            Chunk(
                article_no=article.number,
                article_title=article.title,
                part_no=None,
                anchor=article.anchor,
                text=f"{header}\n{body}".strip(),
            )
        ]

    pieces: list[tuple[str | None, str]] = []
    for point_no, part in _split_by_points(article.text):
        part = _normalize(part)
        if not part:
            continue
        if len(part) + len(header) > max_chars:
            # Пункт делится на фрагменты предложениями, но все они —
            # один и тот же пункт, поэтому номер не меняется.
            pieces.extend(
                (point_no, fragment)
                for fragment in _hard_split(part, max_chars - len(header))
            )
        else:
            pieces.append((point_no, part))

    # Слишком короткие куски приклеиваем к предыдущему — отдельно они бесполезны.
    # Номер пункта берётся от того, к кому приклеиваем, а не от приклеиваемого.
    merged: list[tuple[str | None, str]] = []
    for point_no, piece in pieces:
        if merged and len(piece) < min_chars:
            prev_no, prev_text = merged[-1]
            merged[-1] = (prev_no, f"{prev_text} {piece}")
        else:
            merged.append((point_no, piece))

    chunks = []
    counter = 0
    for point_no, piece in merged:
        if point_no is None:
            counter += 1
            part_no = str(counter)
        else:
            part_no = point_no
        chunks.append(
            Chunk(
                article_no=article.number,
                article_title=article.title,
                part_no=part_no,
                anchor=article.anchor,
                text=f"{header}\n{piece}",
            )
        )
    return chunks


def chunk_act(act: Act, max_chars: int = 2000, min_chars: int = 100) -> list[Chunk]:
    chunks: list[Chunk] = []
    for article in act.articles:
        chunks.extend(chunk_article(article, max_chars, min_chars))
    return chunks
