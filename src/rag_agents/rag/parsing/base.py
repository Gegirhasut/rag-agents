"""Контракты парсеров (ARCHITECTURE §5, §7.2). Парсер отдаёт секции генератором."""

from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel

from rag_agents.domain.enums import SourceFormat

TocSource = Literal["native", "heuristic", "none"]


class Block(BaseModel):
    """Абзац, строка диалога, строфа или элемент списка."""

    text: str
    page: int | None = None  # PDF: 1-based


class Section(BaseModel):
    path: list[str]  # ["Часть первая", "Глава IV"]
    title: str | None
    level: int
    blocks: list[Block]


class ParsedMeta(BaseModel):
    """Метаданные документа. toc_source и sections парсер дописывает по ходу итерации:
    структура известна только после прохода по всему файлу."""

    title: str | None
    author: str | None
    language: str | None = None
    year: str | None = None
    encoding: str | None = None
    pages: int | None = None
    toc_source: TocSource = "none"


class Parser(Protocol):
    format: SourceFormat

    def parse(self, path: Path, filename: str) -> tuple[ParsedMeta, Iterator[Section]]: ...
