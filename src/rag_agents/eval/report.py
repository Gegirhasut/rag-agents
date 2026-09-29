"""Markdown-отчёты: прогон (reports/eval/<date>_<config>.md) и сравнение двух прогонов."""

from collections.abc import Sequence
from typing import Any

from rag_agents.domain.eval import EvalItemResult, EvalRunOut
from rag_agents.eval.metrics import HEADLINE, quality, worst
from rag_agents.eval.stats import Diff

# Порядок строк сводной таблицы; всё прочее из metrics — ниже в алфавитном порядке
ORDER = (
    "hit@5",
    "hit@8",
    "hit@20",
    "recall@5",
    "recall@8",
    "recall@20",
    "mrr@10",
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "key_facts_coverage",
    "citation_validity",
    "citation_support",
    "has_citation",
    "refusal_correct",
    "empty_answer",
    "injection_resisted",
    "max_dense_score",
)
_ANSWER_CHARS = 500


def _f(v: Any, digits: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _one_line(s: str | None, limit: int) -> str:
    text = " ".join((s or "").split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_run(run: EvalRunOut, items: Sequence[EvalItemResult]) -> str:
    m = run.metrics or {}
    cfg = run.config
    llm, eval_cfg = cfg.get("llm", {}), cfg.get("eval", {})
    lines = [
        f"# Eval: {run.config_name} — {run.started_at:%Y-%m-%d %H:%M} UTC",
        "",
        f"- Прогон: `{run.id}`",
        f"- Датасет: `{run.dataset}` (sha `{run.dataset_sha}`), вопросов: {m.get('questions')}, "
        f"ошибок: {m.get('errors')}",
        f"- LLM: `{llm.get('model')}` (effort `{llm.get('reasoning_effort')}`), "
        f"эмбеддинги: `{cfg.get('embedding_model')}`, промпт: `{cfg.get('prompt_version')}`",
        f"- Retrieval: `{cfg.get('retrieval')}`, кандидатов для метрик: {eval_cfg.get('search_k')}",
        f"- Судья: `{cfg.get('judge')}`, git: `{cfg.get('git_sha')}`",
        "",
        "## Сводка",
        "",
        "| Метрика | Среднее | n |",
        "|---|---:|---:|",
    ]
    metrics: dict[str, dict[str, Any]] = m.get("metrics", {})
    names = [k for k in ORDER if k in metrics] + sorted(k for k in metrics if k not in ORDER)
    lines += [f"| {k} | {_f(metrics[k]['mean'])} | {metrics[k]['n']} |" for k in names]

    refusal = m.get("refusal", {})
    lat, ttft, cost = m.get("latency_ms", {}), m.get("ttft_ms", {}), m.get("cost_usd", {})
    lines += [
        "",
        f"**Отказы:** precision {_f(refusal.get('precision'))}, "
        f"recall {_f(refusal.get('recall'))} (верных отказов {refusal.get('tp')}, "
        f"лишних {refusal.get('fp')}, пропущенных {refusal.get('fn')})",
        "",
        f"**Латентность:** ответ p50 {_f(lat.get('p50'))} мс / p95 {_f(lat.get('p95'))} мс; "
        f"первый токен p50 {_f(ttft.get('p50'))} мс / p95 {_f(ttft.get('p95'))} мс",
        "",
        f"**Стоимость:** ответы ${_f(cost.get('answers'), 4)} "
        f"(${_f(cost.get('per_question'), 5)} на вопрос), судья ${_f(cost.get('judge'), 4)}",
        "",
        "## По категориям",
        "",
    ]
    cats: dict[str, dict[str, Any]] = m.get("by_category", {})
    cols = [*HEADLINE, "max_dense_score"]
    lines += [
        "| Категория | n | " + " | ".join(cols) + " |",
        "|---|---:|" + "---:|" * len(cols),
    ]
    lines += [
        f"| {cat} | {row['n']} | " + " | ".join(_f(row.get(c), 2) for c in cols) + " |"
        for cat, row in cats.items()
    ]
    lines += ["", "## 10 худших примеров", ""]
    for it in worst(items):
        found = ", ".join(
            f"{'✅' if r.relevant else '·'} {_one_line(r.book_title, 30)} / "
            f"{' / '.join(r.section_path[-2:])} ({r.score:.3f})"
            for r in it.retrieved[:5]
        )
        key_scores = ", ".join(
            f"{k}={_f(it.scores.get(k), 2)}" for k in HEADLINE if it.scores.get(k) is not None
        )
        lines += [
            f"### {it.item_id} · {it.category.value} · качество {quality(it):.2f}",
            "",
            f"**Вопрос:** {it.question}",
            "",
            f"**Ответ:** {_one_line(it.answer or it.error, _ANSWER_CHARS)}",
            "",
            f"**Метрики:** {key_scores or '—'}",
            "",
            f"**Найдено (top-5):** {found or '—'}",
            "",
        ]
        if it.trace_id:
            lines += [f"Трейс: `/insights/traces/{it.trace_id}`", ""]
    return "\n".join(lines).rstrip() + "\n"


def render_diff(a: EvalRunOut, b: EvalRunOut, diffs: Sequence[Diff]) -> str:
    warn = (
        ""
        if a.dataset_sha == b.dataset_sha
        else f"\n> ⚠️ Разные версии датасета: `{a.dataset_sha}` vs `{b.dataset_sha}` — "
        "сравниваются только общие вопросы.\n"
    )
    judge_a, judge_b = a.config.get("judge"), b.config.get("judge")
    if judge_a != judge_b:
        # Разные промпты или модель судьи сдвигают метрики сами по себе (ARCHITECTURE ADR-10)
        warn += (
            f"\n> ⚠️ Разные судьи: `{judge_a}` vs `{judge_b}` — метрики судьи (faithfulness, "
            "relevancy, context_*) несопоставимы, сравнивайте только retrieval.\n"
        )
    lines = [
        f"# Eval diff: {a.config_name} (A) → {b.config_name} (B)",
        "",
        f"A = `{a.id}` ({a.started_at:%Y-%m-%d %H:%M}), "
        f"B = `{b.id}` ({b.started_at:%Y-%m-%d %H:%M})",
        warn,
        "Парный bootstrap по вопросам, 10 000 ресэмплов, 95 % CI разницы B − A. "
        "«лучше» / «хуже» — только если CI не пересекает 0.",
        "",
        "| Метрика | n | A | B | B − A | 95 % CI | Вывод |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    lines += [
        f"| {d.metric} | {d.n} | {d.mean_a:.3f} | {d.mean_b:.3f} | {d.diff:+.3f} | "
        f"[{d.lo:+.3f}; {d.hi:+.3f}] | {d.verdict} |"
        for d in diffs
    ]
    return "\n".join(lines) + "\n"
