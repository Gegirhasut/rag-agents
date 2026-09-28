"""Эвристика заголовков для TXT и PDF без оглавления (ARCHITECTURE §7.4.1).

Уровень — по типу маркера: ТОМ > КНИГА > ЧАСТЬ > ГЛАВА > номер. Меньше число — выше уровень.
"""

import re
from dataclasses import dataclass, field

MAX_HEADING_CHARS = 80
_MONTHS = "января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря"
_KEYWORD = re.compile(
    r"^(?P<kw>том|книга|часть|глава|отдел|раздел|действие|явление)\s+"
    r"(?P<num>[IVXLCDM]+|\d+|[А-Яа-яЁё]+)\b",
    re.IGNORECASE,
)
_ROMAN = re.compile(r"^[IVXLCDM]{1,8}\.?$")
_ARABIC = re.compile(r"^\d{1,3}\.?$")
_DIARY_DATE = re.compile(rf"^\d{{1,2}}\s+({_MONTHS})\s+\d{{4}}", re.IGNORECASE)
_LEVELS = {
    "том": 1,
    "книга": 2,
    "часть": 3,
    "отдел": 3,
    "раздел": 3,
    "действие": 3,
    "глава": 4,
    "явление": 5,
}
NUMBER_LEVEL = 5
CAPS_LEVEL = 4
DIARY_LEVEL = 4


def heading_level(line: str) -> int | None:
    """Уровень заголовка или None, если строка на заголовок не похожа."""
    s = line.strip()
    if not s or len(s) > MAX_HEADING_CHARS or "\n" in s:
        return None
    m = _KEYWORD.match(s)
    if m:
        return _LEVELS[m.group("kw").lower()]
    if _ROMAN.match(s) or _ARABIC.match(s):
        return NUMBER_LEVEL
    if _DIARY_DATE.match(s):
        return DIARY_LEVEL
    letters = [c for c in s if c.isalpha()]
    if (
        len(letters) >= 3  # noqa: PLR2004  «ОН» — не заголовок
        and all(c.isupper() for c in letters)
        and not s.endswith((".", ",", ";", ":"))
        and len(s.split()) <= 8  # noqa: PLR2004  длинная строка прописными — скорее акцент
    ):
        return CAPS_LEVEL
    return None


@dataclass
class HeadingStack:
    """Текущий путь секции: новый заголовок уровня L снимает со стека уровни ≥ L."""

    items: list[tuple[int, str]] = field(default_factory=list)

    def push(self, level: int, title: str) -> None:
        while self.items and self.items[-1][0] >= level:
            self.items.pop()
        self.items.append((level, title))

    def truncate(self, level: int) -> None:
        """Выход из секции уровня level + 1: остаются только уровни ≤ level."""
        while self.items and self.items[-1][0] > level:
            self.items.pop()

    @property
    def path(self) -> list[str]:
        return [t for _, t in self.items]

    @property
    def level(self) -> int:
        return len(self.items)
