"""Парный bootstrap по вопросам: 95 % CI разницы метрики двух прогонов (ARCHITECTURE §15.2).

При ~50 вопросах разница в несколько пунктов бывает шумом. Ресэмплим вопросы (одни и те же
индексы для обоих прогонов — поэтому «парный»), считаем среднюю разницу B − A на каждом
ресэмпле и берём 2.5 и 97.5 перцентили. «Улучшение» — только если CI не пересекает 0.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rag_agents.domain.eval import EvalItemResult

N_RESAMPLES = 10_000
# Метрики, где рост — ухудшение
LOWER_IS_BETTER = frozenset({"empty_answer"})


@dataclass(frozen=True)
class Diff:
    metric: str
    n: int  # вопросов, где метрика определена в обоих прогонах
    mean_a: float
    mean_b: float
    diff: float  # B − A
    lo: float
    hi: float

    @property
    def verdict(self) -> str:
        better, worse = "лучше", "хуже"
        if self.metric in LOWER_IS_BETTER:
            better, worse = worse, better
        if self.lo > 0:
            return better
        if self.hi < 0:
            return worse
        return "шум"


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], *, n_resamples: int = N_RESAMPLES, seed: int = 0
) -> tuple[float, float, float]:
    """(средняя разница B − A, нижняя и верхняя граница 95 % CI)."""
    if len(a) != len(b) or not a:
        raise ValueError("нужны непустые выборки одинаковой длины")
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_resamples, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def compare(
    a: Sequence[EvalItemResult], b: Sequence[EvalItemResult], metrics: Sequence[str]
) -> list[Diff]:
    """Сравнение по общим вопросам (item_id); метрика берётся там, где определена в обоих."""
    by_id = {it.item_id: it for it in a}
    pairs = [(by_id[it.item_id], it) for it in b if it.item_id in by_id]
    out = []
    for m in metrics:
        xs = [
            (pa.scores[m], pb.scores[m])
            for pa, pb in pairs
            if pa.scores.get(m) is not None and pb.scores.get(m) is not None
        ]
        if not xs:
            continue
        va = [float(x) for x, _ in xs if x is not None]
        vb = [float(y) for _, y in xs if y is not None]
        diff, lo, hi = paired_bootstrap(va, vb)
        out.append(Diff(m, len(xs), float(np.mean(va)), float(np.mean(vb)), diff, lo, hi))
    return out
