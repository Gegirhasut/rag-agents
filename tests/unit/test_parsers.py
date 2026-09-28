"""Парсеры FB2, EPUB, PDF, DOCX на маленьких реальных файлах (tests/fixtures/make_fixtures.py)."""

import zipfile
from pathlib import Path

import pytest

from rag_agents.core.errors import PermanentError
from rag_agents.domain.enums import SourceFormat
from rag_agents.rag.parsing.base import ParsedMeta, Section
from rag_agents.rag.parsing.detect import (
    UnsupportedFormatError,
    format_by_name,
    magic_matches,
)
from rag_agents.rag.parsing.registry import parser_for

FIXTURES = Path(__file__).parents[1] / "fixtures"


def parse(name: str) -> tuple[ParsedMeta, list[Section]]:
    meta, sections = parser_for(format_by_name(name)).parse(FIXTURES / name, name)
    return meta, list(sections)


def texts(sections: list[Section]) -> list[str]:
    return [b.text for s in sections for b in s.blocks]


@pytest.mark.parametrize("name", ["tolstoy.fb2", "tolstoy.fb2.zip"])
def test_fb2_structure_meta_notes(name: str) -> None:
    meta, sections = parse(name)
    assert (meta.title, meta.author, meta.language, meta.year) == (
        "Исповедь",
        "Лев Николаевич Толстой",
        "ru",
        "1882",
    )
    assert meta.toc_source == "native"
    assert [s.path for s in sections] == [
        ["Часть первая", "Глава I"],
        ["Часть первая", "Глава II"],
        ["Часть вторая", "Глава III"],
        ["Примечания"],
    ]
    first = [b.text for b in sections[0].blocks]
    assert first[0] == "Эпиграф к главе."
    # Ссылка на сноску → маркер, сама сноска — отдельная секция «Примечания»
    assert first[2].endswith("юности.[прим. 1]")
    assert [b.text for b in sections[3].blocks] == ["[прим. 1] Примечание о крещении."]
    assert "Строка стиха первая" in texts(sections)


def test_fb2_does_not_resolve_external_entities() -> None:
    """XXE: сущность на /etc/passwd в DOCTYPE не раскрывается."""
    _, sections = parse("tolstoy.fb2")
    assert not any("root:" in t for t in texts(sections))


def test_epub_spine_order_toc_titles_and_skips_nonlinear_cover() -> None:
    meta, sections = parse("tolstoy.epub")
    assert (meta.title, meta.author, meta.language) == ("Исповедь", "Лев Толстой", "ru")
    assert meta.toc_source == "native"
    assert [s.path for s in sections] == [["Часть первая"], ["Глава II"]]
    assert texts(sections)[0].startswith("Я был крещён")
    assert "Текст обложки" not in texts(sections)
    # Заголовок главы в тексте не дублируется блоком
    assert "Глава II" not in texts(sections)


def test_pdf_outline_pages_running_headers_and_hyphens() -> None:
    meta, sections = parse("tolstoy.pdf")
    assert (meta.title, meta.author, meta.pages, meta.toc_source) == (
        "Исповедь",
        "Лев Толстой",
        6,
        "native",
    )
    assert [s.path for s in sections] == [["Глава первая"], ["Глава вторая"]]
    all_text = texts(sections)
    assert "Это слово с переносом внутри абзаца." in all_text
    assert not any("Толстой. Исповедь" in t for t in all_text)  # колонтитул
    assert not any(t.strip().isdigit() for t in all_text)  # номера страниц
    assert [b.page for b in sections[1].blocks] == [4, 5, 6]


def test_pdf_scan_without_text_layer_is_permanent() -> None:
    with pytest.raises(PermanentError) as e:
        parse("scan.pdf")
    assert e.value.code == "no_text_layer"


def test_broken_pdf_is_corrupted() -> None:
    with pytest.raises(PermanentError) as e:
        parse("broken.pdf")
    assert e.value.code == "corrupted"


def test_docx_headings_by_style_and_tables() -> None:
    meta, sections = parse("tolstoy.docx")
    assert (meta.title, meta.author, meta.toc_source) == ("Исповедь", "Лев Толстой", "native")
    assert [s.path for s in sections] == [["Глава I"], ["Глава I", "Раздел 1.1"], ["Глава II"]]
    assert "1879 | Начало работы над «Исповедью»" in texts(sections)


def test_zip_bomb_is_rejected(tmp_path: Path) -> None:
    bomb = tmp_path / "bomb.epub"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("big.txt", b"\0" * (50 * 1024 * 1024))  # сжатие > 1000:1
    with pytest.raises(PermanentError) as e:
        parser_for(SourceFormat.EPUB).parse(bomb, "bomb.epub")
    assert e.value.code == "zip_bomb"


@pytest.mark.parametrize(
    ("name", "fmt"),
    [
        ("a.TXT", SourceFormat.TXT),
        ("Война и мир.fb2", SourceFormat.FB2),
        ("Война и мир.fb2.zip", SourceFormat.FB2),
        ("book.epub", SourceFormat.EPUB),
        ("diary.pdf", SourceFormat.PDF),
        ("article.docx", SourceFormat.DOCX),
    ],
)
def test_format_by_name(name: str, fmt: SourceFormat) -> None:
    assert format_by_name(name) == fmt


@pytest.mark.parametrize("name", ["a.doc", "a.zip", "a", "a.exe"])
def test_unsupported_formats(name: str) -> None:
    with pytest.raises(UnsupportedFormatError):
        format_by_name(name)


@pytest.mark.parametrize(
    ("fixture", "fmt", "ok"),
    [
        ("tolstoy.pdf", SourceFormat.PDF, True),
        ("tolstoy.epub", SourceFormat.EPUB, True),
        ("tolstoy.docx", SourceFormat.DOCX, True),
        ("tolstoy.fb2", SourceFormat.FB2, True),
        ("tolstoy.fb2.zip", SourceFormat.FB2, True),
        ("chapters_cp1251.txt", SourceFormat.TXT, True),
        ("tolstoy.pdf", SourceFormat.TXT, False),  # бинарник под видом txt
        ("tolstoy.docx", SourceFormat.PDF, False),
        ("chapters_cp1251.txt", SourceFormat.EPUB, False),
    ],
)
def test_magic_bytes(fixture: str, fmt: SourceFormat, ok: bool) -> None:
    assert magic_matches(fmt, (FIXTURES / fixture).read_bytes()[:8192]) is ok
