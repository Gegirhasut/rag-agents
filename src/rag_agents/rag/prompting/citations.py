"""Разбор ссылок [n] в ответе LLM: валидность цитат (ARCHITECTURE §7.8, метрики eval)."""

import re
from dataclasses import dataclass

# [3], [1, 2], [1,2,5]; сноски вида [прим. 1] и годы в скобках сюда не попадают
_MARKER = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")


@dataclass(frozen=True)
class CitationCheck:
    markers: int  # всего ссылок в ответе (с повторами)
    valid: int  # из них указывают на существующий источник 1..n_sources
    used: frozenset[int]  # номера валидных источников, на которые есть ссылки

    @property
    def validity(self) -> float | None:
        """Доля валидных ссылок; None — ссылок нет (метрика не определена)."""
        return self.valid / self.markers if self.markers else None


def cited_numbers(answer: str) -> list[int]:
    """Все номера из ссылок [n] и [n, m] в порядке появления."""
    return [int(n) for m in _MARKER.finditer(answer) for n in m.group(1).split(",")]


def check_citations(answer: str, n_sources: int) -> CitationCheck:
    numbers = cited_numbers(answer)
    valid = [n for n in numbers if 1 <= n <= n_sources]
    return CitationCheck(markers=len(numbers), valid=len(valid), used=frozenset(valid))


def is_grounded(answer: str, refused: bool, n_sources: int) -> bool:
    """grounded = отказ или хотя бы одна валидная ссылка на источник (§7.8)."""
    return refused or check_citations(answer, n_sources).valid > 0
