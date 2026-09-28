import pytest

from rag_agents.rag.cleaning.normalize import clean_text, is_junk, is_scene_break
from rag_agents.rag.cleaning.orthography import norm_text
from rag_agents.rag.cleaning.pdf_artifacts import detect_running_lines, is_page_number, join_lines
from rag_agents.rag.parsing.headings import HeadingStack, heading_level


def test_clean_text_removes_invisible_and_collapses_spaces() -> None:
    raw = "При­мер\u200b  текста с\tNBSP﻿ \r\n строка"
    assert clean_text(raw) == "Пример текста с NBSP\nстрока"


def test_clean_text_keeps_quotes_and_dashes() -> None:
    assert clean_text("«Жизнь — это…»") == "«Жизнь — это…»"


@pytest.mark.parametrize(("text", "junk"), [("12", True), ("* * *", True), ("— 5 —", True),
                                             ("Глава", False), ("I.", False)])  # fmt: skip
def test_junk_blocks(text: str, junk: bool) -> None:
    assert is_junk(text) is junk


def test_scene_break() -> None:
    assert is_scene_break("* * *")
    assert not is_scene_break("Так было.")


@pytest.mark.parametrize(
    ("src", "norm"),
    [
        ("Ёлка и её", "Елка и ее"),
        ("мiръ", "мир"),
        ("вѣра", "вера"),
        ("Ѳеодоръ", "Феодор"),
        ("міръ", "мир"),
        ("съел", "съел"),  # ъ не на конце слова — не трогаем
    ],
)
def test_norm_text_modernizes_orthography(src: str, norm: str) -> None:
    assert norm_text(src) == norm


def test_join_lines_dehyphenates_only_lowercase_cyrillic() -> None:
    assert join_lines("сло-\nво и строка\nдальше") == "слово и строка дальше"
    assert join_lines("Толстой-\nПисатель") == "Толстой- Писатель"
    assert join_lines("XIX-\nго") == "XIX- го"


def test_running_lines_detected_with_page_numbers_normalized() -> None:
    pages = [
        ["Л. Н. Толстой. Исповедь", f"Абзац {i}", "середина", f"Текст {i}", "конец", f"Стр. {i}"]
        for i in range(10)
    ]
    running = detect_running_lines(pages)
    assert "л. н. толстой. исповедь" in running
    assert "стр. #" in running
    assert "текст #" not in running  # не в первых/последних двух строках — не колонтитул
    assert detect_running_lines(pages[:2]) == set()  # мало страниц — не решаем


def test_page_number() -> None:
    assert is_page_number(" 12 ")
    assert is_page_number("— 7 —")
    assert not is_page_number("12 глава")


@pytest.mark.parametrize(
    ("line", "level"),
    [
        ("ТОМ ВТОРОЙ", 1),
        ("Часть третья", 3),
        ("ГЛАВА XIV", 4),
        ("Глава 5", 4),
        ("XIV", 5),
        ("12.", 5),
        ("ИСПОВЕДЬ", 4),
        ("12 марта 1851", 4),
        ("Я был крещён.", None),
        ("ОН", None),
        ("x" * 90, None),
    ],
)
def test_heading_level(line: str, level: int | None) -> None:
    assert heading_level(line) == level


def test_heading_stack_nesting() -> None:
    st = HeadingStack()
    st.push(3, "Часть I")
    st.push(4, "Глава 1")
    assert st.path == ["Часть I", "Глава 1"]
    st.push(4, "Глава 2")
    assert st.path == ["Часть I", "Глава 2"]
    st.push(3, "Часть II")
    assert st.path == ["Часть II"]
    st.truncate(0)
    assert st.path == []
