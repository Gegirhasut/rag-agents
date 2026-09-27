from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel

from rag_agents.domain.enums import SourceFormat


class Block(BaseModel):
    """Абзац или строка диалога."""

    text: str
    page: int | None = None


class Section(BaseModel):
    path: list[str]
    title: str | None
    level: int
    blocks: list[Block]


class ParsedMeta(BaseModel):
    title: str | None
    author: str | None
    encoding: str | None = None
    toc_source: Literal["native", "heuristic", "none"] = "none"


class Parser(Protocol):
    format: SourceFormat

    def parse(self, path: Path, filename: str) -> tuple[ParsedMeta, Iterator[Section]]: ...
