from pathlib import Path

import pytest

from rag_agents.core.errors import PermanentError
from rag_agents.rag.parsing.txt import TxtParser, detect_encoding, meta_from_filename

TEXT = (
    "Я был крещён и воспитан в православной христианской вере.\n"
    "Меня учили ей и с детства.\n\n"
    "Судя по некоторым воспоминаниям, я никогда и не верил серьёзно.\n"
)


@pytest.mark.parametrize("encoding", ["utf-8", "cp1251", "koi8_r", "utf-8-sig"])
def test_parses_russian_text_in_common_encodings(tmp_path: Path, encoding: str) -> None:
    f = tmp_path / "book.txt"
    f.write_bytes(TEXT.encode(encoding))

    meta, sections = TxtParser().parse(f, "book.txt")
    [section] = list(sections)

    assert meta.encoding == encoding
    assert [b.text for b in section.blocks] == [
        "Я был крещён и воспитан в православной христианской вере. Меня учили ей и с детства.",
        "Судя по некоторым воспоминаниям, я никогда и не верил серьёзно.",
    ]


def test_utf8_multibyte_char_cut_by_sample_boundary_is_still_utf8() -> None:
    # 1 байт + двухбайтные «ж»: граница сэмпла (64 КБ) приходится на середину символа
    data = b"a" + ("ж" * 40_000).encode("utf-8")
    assert data[: 64 * 1024][-1:] == "ж".encode()[:1]
    assert detect_encoding(data) == "utf-8"


def test_text_without_blank_lines_splits_by_lines(tmp_path: Path) -> None:
    f = tmp_path / "lines.txt"
    f.write_bytes("Первая строка\r\nВторая  строка\n".encode())
    _, sections = TxtParser().parse(f, "lines.txt")
    assert [b.text for b in next(sections).blocks] == ["Первая строка", "Вторая строка"]


def test_chapters_are_detected_by_heading_heuristic() -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "chapters_cp1251.txt"
    meta, sections = TxtParser().parse(fixture, "Л. Н. Толстой - Исповедь.txt")
    result = [(s.path, len(s.blocks)) for s in sections]
    assert result == [(["ГЛАВА I"], 2), (["ГЛАВА II"], 2)]  # «* * *» — не блок
    assert (meta.encoding, meta.toc_source) == ("cp1251", "heuristic")
    assert (meta.author, meta.title) == ("Л. Н. Толстой", "Исповедь")


def test_streaming_decode_across_read_chunks(tmp_path: Path) -> None:
    """Файл больше буфера чтения (1 МБ): многобайтные символы на границах не бьются."""
    para = "Жизнь моя остановилась, ё и ѣ. " * 20
    f = tmp_path / "big.txt"
    f.write_text((para + "\n\n") * 3000, encoding="utf-8")
    _, sections = TxtParser().parse(f, "big.txt")
    blocks = [b.text for s in sections for b in s.blocks]
    assert len(blocks) == 3000
    assert all(b == para.strip() for b in blocks)


def test_strips_soft_hyphen_and_zero_width(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("При\u00adмер\u200b текста", encoding="utf-8")
    _, sections = TxtParser().parse(f, "a.txt")
    assert next(sections).blocks[0].text == "Пример текста"


def test_empty_file_is_permanent_error(tmp_path: Path) -> None:
    f = tmp_path / "empty.txt"
    f.write_bytes(b"  \n ")
    with pytest.raises(PermanentError) as e:
        TxtParser().parse(f, "empty.txt")
    assert e.value.code == "empty_document"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("Л. Н. Толстой - Исповедь.txt", ("Л. Н. Толстой", "Исповедь")),
        ("Толстой — Война и мир.txt", ("Толстой", "Война и мир")),
        ("ispoved_cp1251.txt", (None, "ispoved cp1251")),
        ("Толстой - Война и мир.fb2.zip", ("Толстой", "Война и мир")),
    ],
)
def test_meta_from_filename(filename: str, expected: tuple[str | None, str]) -> None:
    assert meta_from_filename(filename) == expected
