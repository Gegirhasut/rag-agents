"""Данные страницы «Качество» (/eval): прогоны eval → таблицы и ряды для графиков.

Чистая презентация поверх EvalService и функций пакета eval (метрики, парный bootstrap):
считается то же, что в markdown-отчёте и `make eval-diff`, только для браузера.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rag_agents.domain.eval import EvalCategory, EvalItemResult, EvalRunOut
from rag_agents.eval.metrics import quality, worst
from rag_agents.eval.report import ORDER
from rag_agents.eval.stats import LOWER_IS_BETTER, compare

# Подписи метрик и группа (цвет на графике): retrieval / генерация / цитаты / отказы
METRICS: dict[str, tuple[str, str]] = {
    "hit@5": ("hit@5", "retrieval"),
    "hit@8": ("hit@8", "retrieval"),
    "hit@20": ("hit@20", "retrieval"),
    "recall@5": ("recall@5", "retrieval"),
    "recall@8": ("recall@8", "retrieval"),
    "recall@20": ("recall@20", "retrieval"),
    "mrr@10": ("MRR@10", "retrieval"),
    "context_precision": ("context precision", "retrieval"),
    "context_recall": ("context recall", "retrieval"),
    "faithfulness": ("faithfulness", "answer"),
    "answer_relevancy": ("answer relevancy", "answer"),
    "key_facts_coverage": ("key facts", "answer"),
    "citation_validity": ("citation validity", "citations"),
    "citation_support": ("citation support", "citations"),
    "has_citation": ("есть цитата", "citations"),
    "refusal_correct": ("верный отказ", "refusal"),
    "empty_answer": ("пустой ответ ↓", "refusal"),
    "injection_resisted": ("инъекция отбита", "refusal"),
}
CATEGORY_LABELS = {
    EvalCategory.FACTUAL: "факты",
    EvalCategory.INTERPRETIVE: "интерпретация",
    EvalCategory.MULTI_HOP: "multi-hop",
    EvalCategory.OUT_OF_CORPUS: "вне корпуса",
    EvalCategory.INJECTION: "инъекция",
}
# Метрики графика «по категориям»: у отказных категорий определён только refusal_correct
CATEGORY_METRICS = ("hit@8", "recall@8", "faithfulness", "answer_relevancy", "refusal_correct")
TREND_METRICS = ("hit@8", "recall@8", "faithfulness", "answer_relevancy", "refusal_correct")
KPI = ("hit@8", "recall@8", "mrr@10", "faithfulness", "answer_relevancy", "refusal_correct")


@dataclass
class DiffRow:
    key: str
    label: str
    n: int
    a: float
    b: float
    diff: float
    lo: float
    hi: float
    verdict: str  # лучше / хуже / шум
    lower_is_better: bool


@dataclass
class EvalPage:
    run: EvalRunOut
    items: list[EvalItemResult]
    runs: list[EvalRunOut]
    base: EvalRunOut | None
    diff: list[DiffRow] = field(default_factory=list)
    charts: dict[str, Any] = field(default_factory=dict)

    @property
    def judge(self) -> str | None:
        judge = self.run.config.get("judge")
        return str(judge) if judge else None

    def metric(self, key: str) -> dict[str, Any] | None:
        m = (self.run.metrics or {}).get("metrics", {}).get(key)
        return m if m and m.get("mean") is not None else None

    @property
    def summary(self) -> dict[str, Any]:
        return self.run.metrics or {}

    def worst(self, n: int = 10) -> list[tuple[EvalItemResult, float]]:
        return [(it, quality(it)) for it in worst(self.items, n)]


def run_headline(run: EvalRunOut) -> dict[str, float | None]:
    """Ключевые средние прогона для таблицы прогонов."""
    metrics = (run.metrics or {}).get("metrics", {})
    return {k: (metrics.get(k) or {}).get("mean") for k in KPI}


def default_base(run: EvalRunOut, runs: Sequence[EvalRunOut]) -> EvalRunOut | None:
    """База для сравнения: предыдущий завершённый прогон на том же датасете."""
    older = [
        r
        for r in runs
        if r.finished_at is not None
        and r.dataset == run.dataset
        and r.started_at < run.started_at
        and r.id != run.id
    ]
    return max(older, key=lambda r: r.started_at) if older else None


async def build_page(
    run: EvalRunOut,
    items: list[EvalItemResult],
    runs: list[EvalRunOut],
    base: tuple[EvalRunOut, list[EvalItemResult]] | None,
) -> EvalPage:
    page = EvalPage(run=run, items=items, runs=runs, base=base[0] if base else None)
    if base is not None:
        keys = [k for k in ORDER if k in METRICS]
        # 10 000 ресэмплов × ~18 метрик на numpy — десятые доли секунды, но не в event loop
        diffs = await asyncio.to_thread(compare, base[1], items, keys)
        page.diff = [
            DiffRow(
                key=d.metric,
                label=METRICS[d.metric][0],
                n=d.n,
                a=d.mean_a,
                b=d.mean_b,
                diff=d.diff,
                lo=d.lo,
                hi=d.hi,
                verdict=d.verdict,
                lower_is_better=d.metric in LOWER_IS_BETTER,
            )
            for d in diffs
        ]
    page.charts = {
        "metrics": _metric_bars(page),
        "categories": _categories(run),
        "threshold": _threshold(items),
        "trend": _trend(run, runs),
        "diff": [
            {"label": r.label, "diff": r.diff, "lo": r.lo, "hi": r.hi, "verdict": r.verdict}
            for r in page.diff
        ],
        "latency": _latency(items),
    }
    return page


def _metric_bars(page: EvalPage) -> list[dict[str, Any]]:
    out = []
    for key in ORDER:
        if key not in METRICS:
            continue
        m = page.metric(key)
        if m is not None:
            label, group = METRICS[key]
            out.append({"key": key, "label": label, "group": group, "mean": m["mean"], "n": m["n"]})
    return out


def _categories(run: EvalRunOut) -> dict[str, Any]:
    by_cat = (run.metrics or {}).get("by_category", {})
    cats = [c for c in EvalCategory if c.value in by_cat]
    return {
        "labels": [f"{CATEGORY_LABELS[c]} ({by_cat[c.value]['n']})" for c in cats],
        "series": [
            {
                "key": k,
                "label": METRICS[k][0],
                "data": [by_cat[c.value].get(k) for c in cats],
            }
            for k in CATEGORY_METRICS
        ],
    }


def _threshold(items: Sequence[EvalItemResult]) -> list[dict[str, Any]]:
    """Лучший dense-score вопроса: где «в корпусе» и «вне корпуса» перекрываются (сырьё τ)."""
    return [
        {
            "id": it.item_id,
            "score": it.scores["max_dense_score"],
            "expects_refusal": it.category.expects_refusal,
            "refused": bool(it.refused),
            "category": CATEGORY_LABELS[it.category],
        }
        for it in items
        if it.scores.get("max_dense_score") is not None
    ]


def _trend(run: EvalRunOut, runs: Sequence[EvalRunOut]) -> dict[str, Any]:
    # Только полные прогоны того же датасета: пробные (LIMIT=5) дают шум, а не динамику
    size = (run.metrics or {}).get("questions")
    same = sorted(
        (
            r
            for r in runs
            if r.finished_at
            and r.dataset == run.dataset
            and r.metrics
            and r.metrics.get("questions") == size
        ),
        key=lambda r: r.started_at,
    )
    return {
        "runs": [
            {
                "id": str(r.id),
                "ts": r.started_at.isoformat(),
                "config": r.config_name,
                "current": r.id == run.id,
            }
            for r in same
        ],
        "series": [
            {"key": k, "label": METRICS[k][0], "data": [run_headline(r).get(k) for r in same]}
            for k in TREND_METRICS
        ],
    }


def _latency(items: Sequence[EvalItemResult]) -> list[dict[str, Any]]:
    """Время ответа и до первого токена по вопросам, отсортировано по полному времени."""
    rows = [it for it in items if it.latency_ms is not None]
    rows.sort(key=lambda it: it.latency_ms or 0)
    return [
        {"id": it.item_id, "total": it.latency_ms, "ttft": it.ttft_ms, "refused": bool(it.refused)}
        for it in rows
    ]
