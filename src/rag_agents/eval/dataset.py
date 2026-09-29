import hashlib
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from rag_agents.domain.eval import EvalCategory, GoldenItem


class DatasetError(ValueError):
    pass


def load_dataset(path: Path) -> tuple[list[GoldenItem], str]:
    """JSONL → вопросы + sha256 файла (12 hex): прогоны сравнимы, только если sha совпадает.

    Пустые строки и строки с `//` пропускаются: в датасете можно оставлять пометки.
    """
    raw = path.read_bytes()
    items: list[GoldenItem] = []
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            items.append(GoldenItem.model_validate_json(line))
        except ValidationError as e:
            raise DatasetError(f"{path}:{line_no}: {e}") from e
    dupes = [i for i, n in Counter(it.id for it in items).items() if n > 1]
    if dupes:
        raise DatasetError(f"{path}: повторяются id {dupes}")
    if not items:
        raise DatasetError(f"{path}: нет вопросов")
    return items, hashlib.sha256(raw).hexdigest()[:12]


def category_shares(items: list[GoldenItem]) -> dict[EvalCategory, float]:
    counts = Counter(it.category for it in items)
    return {c: counts[c] / len(items) for c in EvalCategory}
