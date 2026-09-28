"""PDF через PyMuPDF (AGPL, исключение задокументировано в SPEC): постранично, память ~константна.

Структура — outline (get_toc) или эвристика заголовков; колонтитулы и номера страниц
отсекаются детектором повторов (§7.3).
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pymupdf

from rag_agents.core.errors import PermanentError
from rag_agents.domain.enums import SourceFormat
from rag_agents.rag.cleaning.pdf_artifacts import (
    detect_running_lines,
    is_page_number,
    join_lines,
    line_key,
)
from rag_agents.rag.parsing.base import ParsedMeta, Section
from rag_agents.rag.parsing.headings import heading_level
from rag_agents.rag.parsing.sections import SectionBuilder

# Скан: страниц без текстового слоя больше этой доли
NO_TEXT_SHARE = 0.8
# Колонтитулы ищем по выборке страниц: полный лишний проход по 1000-страничному PDF дорог
_SAMPLE_PAGES = 60
_MIN_PAGE_CHARS = 20
_TEXT_BLOCK = 0


def _open(path: Path) -> pymupdf.Document:
    try:
        doc = pymupdf.open(path)
    except (pymupdf.FileDataError, RuntimeError) as e:
        raise PermanentError("corrupted", "Файл повреждён: PDF не открывается") from e
    if doc.needs_pass:
        doc.close()
        raise PermanentError("encrypted", "PDF защищён паролем")
    return doc


def _blocks(page: Any) -> list[str]:
    """Текстовые блоки страницы в порядке чтения (Any: объекты PyMuPDF без типов)."""
    raw = page.get_text("blocks", sort=True)
    return [b[4] for b in raw if b[6] == _TEXT_BLOCK and b[4].strip()]


class PdfParser:
    format = SourceFormat.PDF

    def parse(self, path: Path, filename: str) -> tuple[ParsedMeta, Iterator[Section]]:
        doc = _open(path)
        info = doc.metadata or {}
        meta = ParsedMeta(
            title=(info.get("title") or "").strip() or None,
            author=(info.get("author") or "").strip() or None,
            pages=doc.page_count,
        )
        try:
            running = self._running_lines(doc)
        except Exception:
            doc.close()
            raise
        return meta, self._sections(doc, meta, running)

    @staticmethod
    def _running_lines(doc: pymupdf.Document) -> set[str]:
        n = doc.page_count
        if n == 0:
            raise PermanentError("empty_document", "В PDF нет страниц")
        step = max(1, n // _SAMPLE_PAGES)
        sample = range(0, n, step)
        pages: list[list[str]] = []
        empty = 0
        for i in sample:
            text = doc[i].get_text("text")
            if len(text.strip()) < _MIN_PAGE_CHARS:
                empty += 1
            pages.append([x.strip() for x in text.splitlines() if x.strip()])
        if empty / len(sample) > NO_TEXT_SHARE:
            raise PermanentError(
                "no_text_layer", "В PDF нет текстового слоя (скан): нужен OCR, его пока нет"
            )
        return detect_running_lines(pages)

    def _sections(  # noqa: PLR0912  разметка страницы outline-ом: ветки идут подряд по блокам
        self, doc: pymupdf.Document, meta: ParsedMeta, running: set[str]
    ) -> Iterator[Section]:
        builder = SectionBuilder()
        # outline: [уровень, заголовок, страница (1-based)] → заголовки по страницам
        toc: dict[int, list[tuple[int, str]]] = {}
        for level, title, page in doc.get_toc(simple=True):
            if page >= 1 and str(title).strip():
                toc.setdefault(page, []).append((level, str(title).strip()))
        meta.toc_source = "native" if toc else "none"
        heuristic = 0
        try:
            for i in range(doc.page_count):
                page_no = i + 1
                blocks = [
                    b
                    for b in _blocks(doc[i])
                    if line_key(b) not in running and not is_page_number(b)
                ]
                pending = list(toc.get(page_no, []))
                for raw in blocks:
                    text = join_lines(raw)
                    # Заголовок из outline ставим перед блоком с тем же текстом; не нашли —
                    # в начало страницы (граница главы с точностью до страницы)
                    match = next((t for t in pending if _same(t[1], text)), None)
                    if match is not None:
                        for t in pending[: pending.index(match) + 1]:
                            yield from builder.heading(*t)
                        pending = pending[pending.index(match) + 1 :]
                        continue
                    if pending and not builder.blocks_on_page(page_no):
                        for t in pending:
                            yield from builder.heading(*t)
                        pending = []
                    level = None if toc else heading_level(text)
                    if level is not None:
                        yield from builder.heading(level, text)
                        heuristic += 1
                    else:
                        builder.add(text, page=page_no)
                for t in pending:
                    yield from builder.heading(*t)
            yield from builder.flush()
        finally:
            doc.close()
        if not toc and heuristic:
            meta.toc_source = "heuristic"


def _same(title: str, text: str) -> bool:
    a = " ".join(title.lower().split())
    b = " ".join(text.lower().split())
    return bool(a) and (a == b or (len(b) <= len(a) + 20 and b.startswith(a)))
