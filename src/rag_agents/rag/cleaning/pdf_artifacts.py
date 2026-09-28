"""Артефакты PDF-вёрстки (ARCHITECTURE §7.3, п. 2): переносы, строки, колонтитулы."""

import re
from collections import Counter
from collections.abc import Iterable

# «сло-\nво» → «слово»: только если обе части — кириллица в нижнем регистре,
# иначе ломаются настоящие дефисы («Толстой-\nписатель», «XIX-\nго»)
_HYPHEN_BREAK = re.compile(r"(?<=[а-яё])[-\u00ad\u2010]\n(?=[а-яё])")
_LINE_BREAK = re.compile(r"\s*\n\s*")
_DIGITS = re.compile(r"\d+")
HEADER_SHARE = 0.3
EDGE_LINES = 2


def join_lines(text: str) -> str:
    """Строки внутри PDF-блока — один абзац: склеиваем переносы и строки."""
    text = _HYPHEN_BREAK.sub("", text)
    return _LINE_BREAK.sub(" ", text).strip()


def line_key(line: str) -> str:
    """Ключ повтора: цифры → #, чтобы «Стр. 12» и «Стр. 13» считались одной строкой."""
    return _DIGITS.sub("#", " ".join(line.split())).lower()


def detect_running_lines(pages: Iterable[list[str]], min_pages: int = 4) -> set[str]:
    """Колонтитулы: ключи строк, которые стоят в первых или последних двух строках
    на ≥ 30 % страниц. pages — строки каждой страницы в порядке чтения."""
    counter: Counter[str] = Counter()
    n = 0
    for lines in pages:
        n += 1
        edge = lines[:EDGE_LINES] + lines[-EDGE_LINES:]
        counter.update({line_key(x) for x in edge if x.strip()})
    if n < min_pages:
        return set()
    return {k for k, c in counter.items() if c / n >= HEADER_SHARE}


def is_page_number(line: str) -> bool:
    return bool(re.fullmatch(r"[\s\-–—]*\d{1,4}[\s\-–—]*", line))
