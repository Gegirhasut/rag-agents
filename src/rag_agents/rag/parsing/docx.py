"""DOCX через python-docx: заголовки по стилям, таблицы — строками через « | »."""

import re
from collections.abc import Iterator
from pathlib import Path

import docx
from docx.document import Document as DocxDocument
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph

from rag_agents.core.errors import PermanentError
from rag_agents.domain.enums import SourceFormat
from rag_agents.rag.parsing.base import ParsedMeta, Section
from rag_agents.rag.parsing.safety import open_zip
from rag_agents.rag.parsing.sections import SectionBuilder

_HEADING_STYLE = re.compile(r"^(heading|заголовок)\s*(\d)$", re.IGNORECASE)
_TITLE_STYLES = {"title", "название", "заголовок"}


def style_level(style_name: str | None) -> int | None:
    name = (style_name or "").strip().lower()
    m = _HEADING_STYLE.match(name)
    if m:
        return int(m.group(2))
    return 1 if name in _TITLE_STYLES else None


class DocxParser:
    format = SourceFormat.DOCX

    def parse(self, path: Path, filename: str) -> tuple[ParsedMeta, Iterator[Section]]:
        open_zip(path).close()  # проверка zip-bomb до того, как python-docx распакует всё
        try:
            document = docx.Document(str(path))
        except (PackageNotFoundError, KeyError, ValueError) as e:
            raise PermanentError("corrupted", "Файл повреждён: DOCX не открывается") from e
        props = document.core_properties
        meta = ParsedMeta(
            title=(props.title or "").strip() or None,
            author=(props.author or "").strip() or None,
            language=(props.language or "").strip() or None,
        )
        return meta, self._sections(document, meta)

    @staticmethod
    def _sections(document: DocxDocument, meta: ParsedMeta) -> Iterator[Section]:
        builder = SectionBuilder()
        for item in document.iter_inner_content():
            if isinstance(item, Paragraph):
                level = style_level(item.style.name if item.style else None)
                if level is not None and item.text.strip():
                    yield from builder.heading(level, item.text)
                else:
                    builder.add(item.text)
            elif isinstance(item, Table):
                for row in item.rows:
                    cells: list[str] = []
                    for cell in row.cells:
                        text = " ".join(cell.text.split())
                        # Объединённые ячейки python-docx повторяет — убираем подряд идущие дубли
                        if text and (not cells or cells[-1] != text):
                            cells.append(text)
                    if cells:
                        builder.add(" | ".join(cells))
        yield from builder.flush()
        meta.toc_source = "native" if builder.headings else "none"
