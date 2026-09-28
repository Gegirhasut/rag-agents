"""Сводка «Аналитики» из фактов PG: чистые функции без I/O (ARCHITECTURE §14.5).

Почему не Metrics API Langfuse: на Hobby-тарифе он ограничен 100 запросами в сутки,
а страница обновляется раз в минуту. Всё нужное для сводки (usage, тайминги, оценки,
статусы индексации) и так лежит в нашей БД; Langfuse остаётся для дерева трейса по клику.
"""

import math
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timedelta
from uuid import UUID

from rag_agents.core.observability import trace_id_for
from rag_agents.domain.answers import AnswerUsage
from rag_agents.domain.insights import (
    AgentStats,
    AnswerFact,
    IngestFact,
    InsightsOverview,
    Kpi,
    ModelStats,
    Period,
    StageStats,
    TimePoint,
    TraceRow,
)
from rag_agents.llm.base import LLMUsage
from rag_agents.llm.prices import PriceTable

RECENT_LIMIT = 30
_TITLE_CHARS = 160
# Шаги индексации в порядке пайплайна: ключ в documents.meta.timings → подпись
INGEST_STAGES = (
    ("parse_ms", "parse"),
    ("chunk_ms", "chunk"),
    ("save_chunks_ms", "save_chunks"),
    ("embed_ms", "embed"),
    ("upsert_ms", "upsert"),
)


def percentile(values: Iterable[float], q: float) -> float | None:
    """Перцентиль методом nearest-rank (как «p95» в дашбордах): значение из выборки."""
    data = sorted(values)
    if not data:
        return None
    rank = max(math.ceil(q / 100 * len(data)), 1)
    return data[rank - 1]


def answer_cost(a: AnswerFact, prices: PriceTable) -> float | None:
    """Стоимость ответа: сохранённая при генерации или пересчитанная по таблице цен
    (для ответов, записанных до появления cost_usd)."""
    u = a.usage
    if u is None:
        return None
    if u.cost_usd is not None:
        return u.cost_usd
    if u.provider == "none":
        return 0.0
    usage = LLMUsage(
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cached_input_tokens=u.cached_input_tokens,
    )
    cost = prices.cost(u.provider, u.model, usage, a.created_at)
    return cost.total if cost else None


def _llm(u: AnswerUsage | None) -> bool:
    return u is not None and u.provider != "none"


def _bucket(ts: datetime, period: Period) -> datetime:
    if period.granularity == "hour":
        return ts.replace(minute=0, second=0, microsecond=0)
    return ts.replace(hour=0, minute=0, second=0, microsecond=0)


def _buckets(since: datetime, now: datetime, period: Period) -> list[datetime]:
    step = timedelta(hours=1) if period.granularity == "hour" else timedelta(days=1)
    cur, out = _bucket(since, period), []
    while cur <= now:
        out.append(cur)
        cur += step
    return out


def _stage(trace_name: str, name: str, values: list[float]) -> StageStats | None:
    if not values:
        return None
    return StageStats(
        trace_name=trace_name,
        name=name,
        count=len(values),
        p50_ms=percentile(values, 50),
        p95_ms=percentile(values, 95),
    )


def _stages(answers: list[AnswerFact], docs: list[IngestFact]) -> list[StageStats]:
    usages = [a.usage for a in answers if a.usage is not None]
    llm = [u for u in usages if u.provider != "none"]
    query = [
        _stage("query", "embed_query", [u.t_embed_ms for u in usages if u.t_embed_ms is not None]),
        _stage(
            "query", "qdrant_search", [u.t_search_ms for u in usages if u.t_search_ms is not None]
        ),
        _stage(
            "query",
            "llm: до первого токена",
            [u.t_first_token_ms - u.t_retrieval_ms for u in llm if u.t_first_token_ms is not None],
        ),
        _stage(
            "query",
            "llm: генерация ответа",
            [u.t_total_ms - u.t_first_token_ms for u in llm if u.t_first_token_ms is not None],
        ),
    ]
    ingest = [
        _stage("ingest", label, [d.timings[key] for d in docs if key in d.timings])
        for key, label in INGEST_STAGES
    ]
    return [s for s in (*query, *ingest) if s is not None]


def _agents(answers: list[AnswerFact], prices: PriceTable) -> list[AgentStats]:
    by_agent: dict[tuple[UUID, str], list[AnswerFact]] = defaultdict(list)
    for a in answers:
        by_agent[(a.agent_id, a.agent_name)].append(a)
    stats = []
    for (agent_id, name), items in by_agent.items():
        usages = [a.usage for a in items if a.usage is not None]
        rated = [a.feedback for a in items if a.feedback is not None]
        stats.append(
            AgentStats(
                name=name,
                agent_id=agent_id,
                questions=len(items),
                cost_usd=sum(c for a in items if (c := answer_cost(a, prices)) is not None),
                tokens=sum(u.input_tokens + u.output_tokens for u in usages),
                latency_p50_ms=percentile((u.t_total_ms for u in usages), 50),
                latency_p95_ms=percentile((u.t_total_ms for u in usages), 95),
                ttft_p50_ms=percentile(
                    (u.t_first_token_ms for u in usages if u.t_first_token_ms is not None), 50
                ),
                feedback_count=len(rated),
                feedback_up_rate=sum(1 for f in rated if f > 0) / len(rated) if rated else None,
            )
        )
    return sorted(stats, key=lambda s: (-s.questions, s.name))


def _models(answers: list[AnswerFact], prices: PriceTable) -> list[ModelStats]:
    by_model: dict[str, list[AnswerFact]] = defaultdict(list)
    for a in answers:
        if a.usage is not None and _llm(a.usage):
            by_model[f"{a.usage.provider}/{a.usage.model}"].append(a)
    return sorted(
        (
            ModelStats(
                model=model,
                calls=len(items),
                cost_usd=sum(c for a in items if (c := answer_cost(a, prices)) is not None),
                input_tokens=sum(a.usage.input_tokens for a in items if a.usage),
                output_tokens=sum(a.usage.output_tokens for a in items if a.usage),
            )
            for model, items in by_model.items()
        ),
        key=lambda m: -m.calls,
    )


def _recent(
    answers: list[AnswerFact], docs: list[IngestFact], prices: PriceTable
) -> list[TraceRow]:
    rows = [
        TraceRow(
            trace_id=a.trace_id,
            name="query",
            started_at=a.created_at,
            latency_ms=a.usage.t_total_ms if a.usage else None,
            level="ERROR" if a.status == "error" else "DEFAULT",
            status_message="Ответ не получен (ошибка LLM или поиска)"
            if a.status == "error"
            else None,
            agent=a.agent_name,
            session_id=str(a.chat_id),
            title=(a.question or "—")[:_TITLE_CHARS],
            cost_usd=answer_cost(a, prices),
            tokens=(a.usage.input_tokens + a.usage.output_tokens) if a.usage else None,
            ttft_ms=a.usage.t_first_token_ms if a.usage else None,
            feedback=None if a.feedback is None else a.feedback > 0,
        )
        for a in answers[:RECENT_LIMIT]
    ] + [
        TraceRow(
            # Первая попытка ingest: трейс детерминирован от document_id (ADR-9); все попытки —
            # на странице сессии document-{id}
            trace_id=trace_id_for(f"ingest:{d.document_id}:0"),
            name="ingest",
            started_at=d.created_at,
            latency_ms=d.duration_ms,
            level="ERROR" if d.status == "failed" else "DEFAULT",
            status_message=d.error_message if d.status == "failed" else None,
            agent=d.agent_name,
            session_id=f"document-{d.document_id}",
            title=d.filename,
        )
        for d in docs[:RECENT_LIMIT]
    ]
    rows.sort(key=lambda r: r.started_at, reverse=True)
    return rows[:RECENT_LIMIT]


def build_overview(
    period: Period,
    now: datetime,
    answers: list[AnswerFact],
    docs: list[IngestFact],
    prices: PriceTable,
    project_url: str | None,
) -> InsightsOverview:
    since = now - period.delta
    usages = [a.usage for a in answers if a.usage is not None]
    llm = [u for u in usages if u.provider != "none"]
    rated = [a.feedback for a in answers if a.feedback is not None]
    costs = [c for a in answers if (c := answer_cost(a, prices)) is not None]

    kpi = Kpi(
        questions=len(answers),
        llm_calls=len(llm),
        ingests=len(docs),
        errors=sum(1 for a in answers if a.status == "error")
        + sum(1 for d in docs if d.status == "failed"),
        cost_usd=sum(costs),
        input_tokens=sum(u.input_tokens for u in usages),
        output_tokens=sum(u.output_tokens for u in usages),
        latency_p50_ms=percentile((u.t_total_ms for u in usages), 50),
        latency_p95_ms=percentile((u.t_total_ms for u in usages), 95),
        ttft_p50_ms=percentile((u.t_first_token_ms for u in llm if u.t_first_token_ms), 50),
        ttft_p95_ms=percentile((u.t_first_token_ms for u in llm if u.t_first_token_ms), 95),
        feedback_count=len(rated),
        feedback_up_rate=sum(1 for f in rated if f > 0) / len(rated) if rated else None,
    )

    per_bucket: dict[datetime, list[AnswerFact]] = defaultdict(list)
    for a in answers:
        per_bucket[_bucket(a.created_at, period)].append(a)
    series = []
    for b in _buckets(since, now, period):
        items = per_bucket.get(b, [])
        series.append(
            TimePoint(
                ts=b,
                questions=len(items),
                cost_usd=sum(c for a in items if (c := answer_cost(a, prices)) is not None),
                ttft_p50_ms=percentile(
                    (
                        a.usage.t_first_token_ms
                        for a in items
                        if a.usage and a.usage.t_first_token_ms
                    ),
                    50,
                ),
            )
        )

    return InsightsOverview(
        period=period,
        generated_at=now,
        kpi=kpi,
        series=series,
        agents=_agents(answers, prices),
        stages=_stages(answers, docs),
        models=_models(answers, prices),
        recent=_recent(answers, docs, prices),
        langfuse_project_url=project_url,
    )
