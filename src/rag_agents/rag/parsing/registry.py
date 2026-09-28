"""Выбор парсера по формату. Импорт ленивый: worker-embed не грузит PyMuPDF и lxml."""

from rag_agents.domain.enums import SourceFormat
from rag_agents.rag.parsing.base import Parser


def parser_for(fmt: SourceFormat) -> Parser:
    match fmt:
        case SourceFormat.TXT:
            from rag_agents.rag.parsing.txt import TxtParser  # noqa: PLC0415

            return TxtParser()
        case SourceFormat.FB2:
            from rag_agents.rag.parsing.fb2 import Fb2Parser  # noqa: PLC0415

            return Fb2Parser()
        case SourceFormat.EPUB:
            from rag_agents.rag.parsing.epub import EpubParser  # noqa: PLC0415

            return EpubParser()
        case SourceFormat.PDF:
            from rag_agents.rag.parsing.pdf import PdfParser  # noqa: PLC0415

            return PdfParser()
        case SourceFormat.DOCX:
            from rag_agents.rag.parsing.docx import DocxParser  # noqa: PLC0415

            return DocxParser()
