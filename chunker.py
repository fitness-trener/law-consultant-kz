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

    Возвращает пары (номер_пункта, текст). Номер — None для не пронумерованного
    вводного текста перед первым пунктом, а также для абзацного и одиночного
    фолбэков, чтобы не выдумывать нумерацию там, где её нет в самом акте
    (нумерация НПА бывает с пропусками из-за исключённых пунктов).
    """
    matches = list(_POINT_RE.finditer(text))
    if matches:
        parts = []
        lead = text[: matches[0].start()]
        if lead.strip():
            # Преамбула перед первым пунктом ("...следующие условия:") —
            # реальный текст статьи, терять его нельзя.
            parts.append((None, lead))
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            parts.append((match.group(1), text[match.start() : end]))
        return parts
    paragraphs = [p for p in text.split("\n") if p.strip()]
    if len(paragraphs) >= 2:
        return [(None, p) for p in paragraphs]
    return [(None, text)]


def _split_oversized_sentence(sentence: str, max_chars: int) -> list[str]:
    """Бьёт одно предложение, которое само длиннее бюджета, по словам —
    подпунктовые перечисления внутри одного предложения не редкость в
    казахстанских НПА. Слово длиннее бюджета (патологический случай —
    склеенный текст без пробелов) режется по сырым символам.
    """
    out: list[str] = []
    buf = ""
    for word in sentence.split(" "):
        if len(word) > max_chars:
            if buf:
                out.append(buf)
                buf = ""
            for i in range(0, len(word), max_chars):
                out.append(word[i : i + max_chars])
            continue
        if buf and len(buf) + len(word) + 1 > max_chars:
            out.append(buf)
            buf = word
        else:
            buf = f"{buf} {word}".strip()
    if buf:
        out.append(buf)
    return out


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Последний рубеж: режет по границам предложений в пределах лимита.

    Предложение само может превышать бюджет (длинное перечисление внутри
    одного предложения, абзац вовсе без `.`/`;`) — такое предложение не
    может быть дописано целиком, иначе max_chars перестаёт быть инвариантом,
    и доразбивается через `_split_oversized_sentence`.
    """
    sentences = re.split(r"(?<=[.;])\s+", text)
    out: list[str] = []
    buf = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if buf:
                out.append(buf)
                buf = ""
            out.extend(_split_oversized_sentence(sentence, max_chars))
            continue
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

    if len(header) + 1 + len(body) <= max_chars:
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
        if len(header) + 1 + len(part) > max_chars:
            # Пункт делится на фрагменты предложениями, но все они —
            # один и тот же пункт, поэтому номер не меняется. Бюджет
            # учитывает перевод строки, которым заголовок склеивается с телом.
            pieces.extend(
                (point_no, fragment)
                for fragment in _hard_split(part, max_chars - len(header) - 1)
            )
        else:
            pieces.append((point_no, part))

    # Слишком короткие куски приклеиваем к предыдущему — отдельно они бесполезны.
    # Номер пункта берётся от того, к кому приклеиваем, а не от приклеиваемого.
    # Если склейка не влезает в max_chars, оставляем короткий кусок отдельно —
    # превышение лимита хуже, чем чанк ниже пола min_chars.
    merged: list[tuple[str | None, str]] = []
    for point_no, piece in pieces:
        if merged and len(piece) < min_chars:
            prev_no, prev_text = merged[-1]
            combined = f"{prev_text} {piece}"
            if len(header) + 1 + len(combined) <= max_chars:
                merged[-1] = (prev_no, combined)
            else:
                merged.append((point_no, piece))
        else:
            merged.append((point_no, piece))

    # Первому куску приклеивать было не к чему (он самый первый), поэтому
    # он мог остаться короче min_chars — приклеиваем его вперёд, к следующему,
    # и берём номер пункта у того, к кому приклеили (он теперь и определяет чанк).
    if len(merged) > 1 and len(merged[0][1]) < min_chars:
        _, lead_text = merged[0]
        next_no, next_text = merged[1]
        combined = f"{lead_text} {next_text}"
        if len(header) + 1 + len(combined) <= max_chars:
            merged[1] = (next_no, combined)
            merged.pop(0)

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
