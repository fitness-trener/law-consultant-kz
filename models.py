"""Общие структуры данных."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Article:
    number: str          # "1" или "152-1"
    title: str
    anchor: str          # якорь на adilet, например "z1"
    text: str


@dataclass
class Act:
    key: str
    title: str
    url: str
    doc_type: str
    act_date: str | None = None       # ISO, дата принятия
    act_number: str | None = None     # "414-V"
    revision_date: str | None = None  # ISO, дата последнего изменения из сносок
    articles: list[Article] = field(default_factory=list)


@dataclass
class Chunk:
    article_no: str
    article_title: str
    part_no: str | None
    anchor: str
    text: str

    @property
    def char_len(self) -> int:
        return len(self.text)
