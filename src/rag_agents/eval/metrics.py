"""Метрики eval — чистые функции без I/O (ARCHITECTURE §15.2)."""

import math
from collections.abc import Iterable, Sequence
from statistics import fmean
from typing import Any

from rag_agents.domain.eval import EvalCategory, EvalItemResult, ExpectedSource, RetrievedRef

K_VALUES = (5, 8, 20)
MRR_K = 10

# Латиница, похожая на кириллицу, встречается в OCR и старых fb2 («ЧАСТЬ BTОРАЯ»)
_HOMOGLYPHS = str.maketrans("ABCEHKMOPTXYaceopxy", "АВСЕНКМОРТХУасеорху")

# Ключевые метрики вопроса: по ним отбираются худшие примеры и строится разбивка по категориям
HEADLINE = (
    "hit@8",
    "recall@8",
    "mrr@10",
    "faithfulness",
    "answer_relevancy",
    "key_facts_coverage",
    "context_precision",
    "context_recall",
    "refusal_correct",
    "citation_validity",
)


def norm(s: str) -> str:
    return " ".join(s.translate(_HOMOGLYPHS).casefold().replace("ё", "е").split())


def matches(ref: RetrievedRef, src: ExpectedSource) -> bool:
    """Документ — подстрока названия книги; section — подпоследовательность section_path."""
    if norm(src.document) not in norm(ref.book_title or ""):
        return False
    path = iter(norm(p) for p in ref.section_path)
    return all(any(norm(want) == got for got in path) for want in src.section)


def mark_relevant(
    refs: list[RetrievedRef], expected: Sequence[ExpectedSource]
) -> list[RetrievedRef]:
    return [r.model_copy(update={"relevant": any(matches(r, s) for s in expected)}) for r in refs]


def hit_at_k(refs: Sequence[RetrievedRef], k: int) -> float:
    return 1.0 if any(r.relevant for r in refs[:k]) else 0.0


def recall_at_k(refs: Sequence[RetrievedRef], expected: Sequence[ExpectedSource], k: int) -> float:
    """Доля ожидаемых источников, найденных в top-k (multi_hop: нужны оба)."""
    top = refs[:k]
    return sum(any(matches(r, s) for r in top) for s in expected) / len(expected)


def mrr_at_k(refs: Sequence[RetrievedRef], k: int = MRR_K) -> float:
    for rank, r in enumerate(refs[:k], start=1):
        if r.relevant:
            return 1 / rank
    return 0.0


def retrieval_scores(
    refs: Sequence[RetrievedRef], expected: Sequence[ExpectedSource]
) -> dict[str, float | None]:
    if not expected:
        return {}
    out: dict[str, float | None] = {}
    for k in K_VALUES:
        out[f"hit@{k}"] = hit_at_k(refs, k)
        out[f"recall@{k}"] = recall_at_k(refs, expected, k)
    out[f"mrr@{MRR_K}"] = mrr_at_k(refs)
    return out


def average_precision(relevance: Sequence[bool]) -> float:
    """Context precision в духе RAGAS: насколько релевантные источники стоят выше остальных."""
    hits, total = 0, 0.0
    for rank, rel in enumerate(relevance, start=1):
        if rel:
            hits += 1
            total += hits / rank
    return total / hits if hits else 0.0


def percentile(values: Sequence[float], p: float) -> float | None:
    """Nearest-rank, как в /insights."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def _mean(values: Iterable[float | None]) -> tuple[float | None, int]:
    present = [v for v in values if v is not None]
    return (fmean(present), len(present)) if present else (None, 0)


def aggregate(items: Sequence[EvalItemResult]) -> dict[str, Any]:
    """Сводка прогона: среднее каждой метрики (по вопросам, где она определена) + n,
    разбивка по категориям, отказы, латентности и стоимость."""
    keys = sorted({k for it in items for k in it.scores})
    overall = {}
    for key in keys:
        mean, n = _mean(it.scores.get(key) for it in items)
        overall[key] = {"mean": mean, "n": n}
    by_category: dict[str, dict[str, float | None]] = {}
    for cat in EvalCategory:
        group = [it for it in items if it.category == cat]
        if group:
            by_category[cat.value] = {
                "n": len(group),
                **{
                    k: _mean(it.scores.get(k) for it in group)[0]
                    for k in (*HEADLINE, "max_dense_score")
                },
            }
    latencies = [it.latency_ms for it in items if it.latency_ms is not None]
    ttfts = [it.ttft_ms for it in items if it.ttft_ms is not None]
    costs = [it.cost_usd for it in items if it.cost_usd is not None]
    return {
        "questions": len(items),
        "errors": sum(1 for it in items if it.error),
        "metrics": overall,
        "by_category": by_category,
        "refusal": refusal_confusion(items),
        "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95)},
        "ttft_ms": {"p50": percentile(ttfts, 50), "p95": percentile(ttfts, 95)},
        "cost_usd": {"answers": sum(costs), "per_question": fmean(costs) if costs else None},
    }


def refusal_confusion(items: Sequence[EvalItemResult]) -> dict[str, float | int | None]:
    """Отказ как классификатор «в корпусе нет ответа»: precision и recall отказов."""
    tp = fp = fn = 0
    for it in items:
        if it.refused is None:
            continue
        expected = it.category.expects_refusal
        tp += int(expected and it.refused)
        fp += int(not expected and it.refused)
        fn += int(expected and not it.refused)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def quality(item: EvalItemResult) -> float:
    """Одно число для «худших примеров»: для отказных — верность отказа, иначе среднее ключевых."""
    if item.error:
        return -1.0
    if item.category.expects_refusal:
        return item.scores.get("refusal_correct") or 0.0
    mean, _ = _mean(
        item.scores.get(k)
        for k in ("hit@8", "key_facts_coverage", "faithfulness", "refusal_correct")
    )
    return mean if mean is not None else 0.0


def worst(items: Sequence[EvalItemResult], n: int = 10) -> list[EvalItemResult]:
    return sorted(items, key=lambda it: (quality(it), it.item_id))[:n]
