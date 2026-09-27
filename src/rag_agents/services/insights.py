"""Аналитика из Langfuse у нас в UI (ARCHITECTURE §14.5): KPI, агенты, шаги пайплайна, трейсы.

Все запросы фильтруются по userId = владелец: Langfuse-проект общий для всех пользователей,
а видеть человек должен только своё (тот же принцип изоляции, что и в PG/Qdrant).
Ответы кэшируются в Redis: Metrics API не рассчитан на запрос при каждом открытии страницы.
"""

import asyncio
import json
import re
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog
from redis.asyncio import Redis

from rag_agents.core.db import Database
from rag_agents.core.langfuse_api import LangfuseApiError, LangfuseReader, iso
from rag_agents.domain.chats import FeedbackStat
from rag_agents.domain.insights import (
    AgentStats,
    InsightsOverview,
    Kpi,
    ModelStats,
    Period,
    SearchHit,
    SpanRow,
    StageStats,
    TimePoint,
    TraceDetail,
    TraceRow,
)
from rag_agents.repositories.agents import AgentRepository
from rag_agents.repositories.chats import ChatRepository
from rag_agents.services.errors import NotFoundError

log = structlog.get_logger()

OVERVIEW_TTL_S = 60
TRACE_TTL_S = 600
PROJECT_TTL_S = 24 * 3600
RECENT_LIMIT = 30
# Старые ingest-трейсы называли батчи «embed_batch 3/6»: в статистике шагов их не показываем
_NUMBERED = re.compile(r"\s\d+/\d+$")
# В metadata SDK кладёт служебные поля (scope.*, resourceAttributes.*, public key) — наружу
# отдаём только свои ключи
_DETAIL_KEYS = (
    "top_k",
    "mode",
    "collection",
    "max_score",
    "hits",
    "dim",
    "sections",
    "has_title",
    "chunks",
    "avg_tokens",
    "max_tokens",
    "target_tokens",
    "chunker",
    "rows",
    "batch",
    "batches",
    "batch_size",
    "texts",
    "points",
    "embed_ms",
    "upsert_ms",
    "chunks_total",
    "duration_ms",
    "format",
    "filename",
    "attempt",
    "embedding_model",
    "prompt_version",
    "provider",
    "reasoning_tokens",
    "refused",
    "citations",
    "reason",
    "temperature",
    "reasoning_effort",
)


class InsightsDisabledError(Exception):
    """Langfuse не настроен (нет ключей): страница показывает подсказку."""


class InsightsUnavailableError(Exception):
    """Langfuse API недоступен или вернул ошибку."""


def _f(v: Any) -> float | None:
    """Metrics API отдаёт числа то строкой, то числом, то null."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v: Any) -> int:
    return int(_f(v) or 0)


def _dt(v: str) -> datetime:
    return datetime.fromisoformat(v.replace("Z", "+00:00"))


def _json(v: Any) -> Any:
    """io-поля v2 observations приходят JSON-строкой."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def _agent_tag(tags: list[str] | None) -> str | None:
    return next((t for t in tags or [] if t != "ingest"), None)


def _short(v: Any, limit: int = 120) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return s if len(s) <= limit else s[: limit - 1] + "…"


class InsightsService:
    def __init__(self, reader: LangfuseReader, redis: Redis, db: Database) -> None:
        self.reader = reader
        self.redis = redis
        self.db = db
        # Metrics API на бесплатном тарифе ограничен по частоте: не больше 4 запросов разом
        self._sem = asyncio.Semaphore(4)

    @property
    def enabled(self) -> bool:
        return self.reader.enabled

    async def _call[T](self, aw: Awaitable[T]) -> T:
        async with self._sem:
            try:
                return await aw
            except LangfuseApiError as e:
                log.warning("insights.langfuse_error", error=str(e))
                raise InsightsUnavailableError(str(e)) from e

    # ---------- ссылки в UI Langfuse ----------

    async def project_url(self) -> str | None:
        """URL проекта в UI Langfuse; None, если трейсинг выключен или API недоступен."""
        if not self.enabled:
            return None
        key = "insights:project_id"
        raw = await self.redis.get(key)
        pid = raw if isinstance(raw, str) else None  # клиент Redis с decode_responses=True
        if pid is None:
            try:
                pid = await self._call(self.reader.project_id())
            except InsightsUnavailableError:
                return None
            await self.redis.set(key, pid, ex=PROJECT_TTL_S)
        return f"{self.reader.base_url}/project/{pid}"

    # ---------- обзор ----------

    async def overview(self, owner_id: UUID, period: Period) -> InsightsOverview:
        if not self.enabled:
            raise InsightsDisabledError
        key = f"insights:overview:{owner_id}:{period.value}"
        cached = await self.redis.get(key)
        if cached:
            return InsightsOverview.model_validate_json(cached)
        result = await self._build_overview(owner_id, period)
        await self.redis.set(key, result.model_dump_json(), ex=OVERVIEW_TTL_S)
        return result

    async def _build_overview(self, owner_id: UUID, period: Period) -> InsightsOverview:
        now = datetime.now(UTC)
        since = now - period.delta
        owner = {"column": "userId", "operator": "=", "value": str(owner_id), "type": "string"}
        root = {"column": "isRootObservation", "operator": "=", "value": True, "type": "boolean"}
        not_root = {**root, "value": False}
        gen = {"column": "type", "operator": "=", "value": "GENERATION", "type": "string"}
        query_trace = {"column": "traceName", "operator": "=", "value": "query", "type": "string"}
        errors = {"column": "level", "operator": "=", "value": "ERROR", "type": "string"}

        def q(
            view: str,
            metrics: list[tuple[str, str]],
            filters: list[dict[str, Any]],
            dims: tuple[str, ...] = (),
            time: bool = False,
        ) -> Awaitable[list[dict[str, Any]]]:
            body: dict[str, Any] = {
                "view": view,
                "metrics": [{"measure": m, "aggregation": a} for m, a in metrics],
                "dimensions": [{"field": d} for d in dims],
                "filters": [owner, *filters],
                "fromTimestamp": iso(since),
                "toTimestamp": iso(now),
                "config": {"row_limit": 200},
            }
            if time:
                body["timeDimension"] = {"granularity": period.granularity}
            return self._call(self.reader.metrics(body))

        latency = [("count", "count"), ("latency", "p50"), ("latency", "p95")]
        gen_metrics = [
            ("count", "count"),
            ("totalCost", "sum"),
            ("inputTokens", "sum"),
            ("outputTokens", "sum"),
            ("timeToFirstToken", "p50"),
            ("timeToFirstToken", "p95"),
        ]
        metric_rows, recent = await asyncio.gather(
            asyncio.gather(
                q("observations", latency, [root], ("traceName",)),
                q("observations", [("count", "count")], [root, errors], ("traceName",)),
                q("observations", gen_metrics, [gen]),
                q(
                    "observations",
                    [("count", "count"), ("totalCost", "sum"), ("timeToFirstToken", "p50")],
                    [gen],
                    time=True,
                ),
                q(
                    "observations",
                    [
                        ("count", "count"),
                        ("totalCost", "sum"),
                        ("totalTokens", "sum"),
                        ("timeToFirstToken", "p50"),
                    ],
                    [gen],
                    ("tags",),
                ),
                q("observations", latency, [root, query_trace], ("tags",)),
                q("observations", latency, [not_root], ("traceName", "name")),
                q(
                    "observations",
                    [
                        ("count", "count"),
                        ("totalCost", "sum"),
                        ("inputTokens", "sum"),
                        ("outputTokens", "sum"),
                    ],
                    [gen],
                    ("providedModelName",),
                ),
            ),
            self._recent(owner_id, since),
        )
        (
            roots,
            root_errors,
            gens,
            series,
            by_tag_gen,
            by_tag_root,
            stages,
            models,
        ) = metric_rows

        by_name = {r["traceName"]: r for r in roots}
        err_by_name = {r["traceName"]: _i(r["count_count"]) for r in root_errors}
        g = gens[0] if gens else {}
        # Оценки — из PG (messages.feedback): score-ы Langfuse не связаны с userId и тегами трейса,
        # а наша БД и так источник правды и изолирует данные по владельцу
        feedback = await self._feedback_stats(owner_id, since)
        up, down = sum(f.up for f in feedback), sum(f.down for f in feedback)
        query_row = by_name.get("query", {})
        kpi = Kpi(
            questions=_i(query_row.get("count_count")),
            llm_calls=_i(g.get("count_count")),
            ingests=_i(by_name.get("ingest", {}).get("count_count")),
            errors=sum(err_by_name.values()),
            cost_usd=_f(g.get("sum_totalCost")) or 0.0,
            input_tokens=_i(g.get("sum_inputTokens")),
            output_tokens=_i(g.get("sum_outputTokens")),
            latency_p50_ms=_f(query_row.get("p50_latency")),
            latency_p95_ms=_f(query_row.get("p95_latency")),
            ttft_p50_ms=_f(g.get("p50_timeToFirstToken")),
            ttft_p95_ms=_f(g.get("p95_timeToFirstToken")),
            feedback_count=up + down,
            feedback_up_rate=up / (up + down) if up + down else None,
        )
        return InsightsOverview(
            period=period,
            generated_at=now,
            kpi=kpi,
            series=[
                TimePoint(
                    ts=_dt(r["time_dimension"]),
                    questions=_i(r.get("count_count")),
                    cost_usd=_f(r.get("sum_totalCost")) or 0.0,
                    ttft_p50_ms=_f(r.get("p50_timeToFirstToken")),
                )
                for r in series
            ],
            agents=await self._agents(owner_id, by_tag_gen, by_tag_root, feedback),
            stages=sorted(
                (
                    StageStats(
                        trace_name=r["traceName"],
                        name=r["name"],
                        count=_i(r["count_count"]),
                        p50_ms=_f(r.get("p50_latency")),
                        p95_ms=_f(r.get("p95_latency")),
                    )
                    for r in stages
                    if r.get("traceName") in {"query", "ingest"}
                    and not _NUMBERED.search(r.get("name") or "")
                ),
                key=lambda s: (s.trace_name != "query", -(s.p50_ms or 0)),
            ),
            models=[
                ModelStats(
                    model=r.get("providedModelName") or "—",
                    calls=_i(r["count_count"]),
                    cost_usd=_f(r.get("sum_totalCost")) or 0.0,
                    input_tokens=_i(r.get("sum_inputTokens")),
                    output_tokens=_i(r.get("sum_outputTokens")),
                )
                for r in models
            ],
            recent=recent,
            langfuse_project_url=await self.project_url(),
        )

    async def _agents(
        self,
        owner_id: UUID,
        gen: list[dict[str, Any]],
        roots: list[dict[str, Any]],
        feedback: list[FeedbackStat],
    ) -> list[AgentStats]:
        """Строки Metrics API сгруппированы по tags (имя агента) — сводим в одну таблицу."""
        ids = await self._agent_ids(owner_id)
        stats: dict[str, AgentStats] = {}

        def row(tags: list[str] | None) -> AgentStats | None:
            name = _agent_tag(tags)
            if name is None:
                return None
            return stats.setdefault(name, AgentStats(name=name, agent_id=ids.get(name)))

        for r in gen:
            if a := row(r.get("tags")):
                a.cost_usd = _f(r.get("sum_totalCost")) or 0.0
                a.tokens = _i(r.get("sum_totalTokens"))
                a.ttft_p50_ms = _f(r.get("p50_timeToFirstToken"))
        for r in roots:
            if a := row(r.get("tags")):
                a.questions = _i(r.get("count_count"))
                a.latency_p50_ms = _f(r.get("p50_latency"))
                a.latency_p95_ms = _f(r.get("p95_latency"))
        names = {agent_id: name for name, agent_id in ids.items()}
        for f in feedback:
            name = names.get(f.agent_id)
            if name is not None and f.up + f.down:
                a = stats.setdefault(name, AgentStats(name=name, agent_id=f.agent_id))
                a.feedback_count = f.up + f.down
                a.feedback_up_rate = f.up / a.feedback_count
        return sorted(stats.values(), key=lambda a: (-a.questions, a.name))

    async def _agent_ids(self, owner_id: UUID) -> dict[str, UUID]:
        """Имя агента (тег трейса) → id, чтобы из таблицы вести на страницу агента."""
        async with self.db.session() as s:
            items = await AgentRepository(s).list_for_owner(owner_id)
        return {i.agent.name: i.agent.id for i in items}

    async def _feedback_stats(self, owner_id: UUID, since: datetime) -> list[FeedbackStat]:
        async with self.db.session() as s:
            return await ChatRepository(s).feedback_stats(owner_id, since)

    async def _feedback_by_trace(self, owner_id: UUID, trace_ids: list[str]) -> dict[str, int]:
        async with self.db.session() as s:
            return await ChatRepository(s).feedback_by_trace(owner_id, trace_ids)

    # ---------- списки трейсов ----------

    async def _recent(
        self, owner_id: UUID, since: datetime, session_id: str | None = None
    ) -> list[TraceRow]:
        roots, gens = await asyncio.gather(
            self._call(
                self.reader.observations(
                    isRootObservation=True,
                    userId=str(owner_id),
                    sessionId=session_id,
                    fromStartTime=since,
                    limit=RECENT_LIMIT if session_id is None else 100,
                    fields="core,basic,time,io,trace_context",
                )
            ),
            self._call(
                self.reader.observations(
                    type="GENERATION",
                    userId=str(owner_id),
                    sessionId=session_id,
                    fromStartTime=since,
                    limit=100,
                    fields="core,basic,usage,metrics",
                )
            ),
        )
        gen_by_trace = {g["traceId"]: g for g in gens}
        fb_by_trace = await self._feedback_by_trace(owner_id, [o["traceId"] for o in roots])
        rows = []
        for o in roots:
            g = gen_by_trace.get(o["traceId"], {})
            usage = g.get("usageDetails") or {}
            rows.append(
                TraceRow(
                    trace_id=o["traceId"],
                    name=o.get("traceName") or o["name"],
                    started_at=_dt(o["startTime"]),
                    latency_ms=_f(o.get("latency")) and (_f(o.get("latency")) or 0) * 1000,
                    level=o.get("level") or "DEFAULT",
                    status_message=o.get("statusMessage") or None,
                    agent=_agent_tag(o.get("tags")),
                    session_id=o.get("sessionId") or None,
                    title=self._title(o),
                    cost_usd=_f(g.get("totalCost")),
                    tokens=_i(usage.get("total")) or None,
                    ttft_ms=(_f(g.get("timeToFirstToken")) or 0) * 1000 or None,
                    feedback=(fb_by_trace[o["traceId"]] > 0)
                    if o["traceId"] in fb_by_trace
                    else None,
                )
            )
        rows.sort(key=lambda r: r.started_at, reverse=True)
        return rows

    @staticmethod
    def _title(o: dict[str, Any]) -> str:
        data = _json(o.get("input"))
        if isinstance(data, dict):
            return str(data.get("filename") or data.get("document_id") or o["name"])
        return _short(data or o["name"], 160)

    async def session(self, owner_id: UUID, session_id: str) -> list[TraceRow]:
        """Все трейсы сессии: чат (session = chat_id) или документ (document-{id})."""
        if not self.enabled:
            raise InsightsDisabledError
        epoch = datetime(2026, 1, 1, tzinfo=UTC)
        return await self._recent(owner_id, epoch, session_id=session_id)

    # ---------- один трейс ----------

    async def trace(self, owner_id: UUID, trace_id: str) -> TraceDetail | None:
        """Дерево span-ов трейса. None — Langfuse ещё не обработал трейс (ingestion асинхронный).

        Чужой трейс → NotFoundError, как и для любых данных другого владельца.
        """
        if not self.enabled:
            raise InsightsDisabledError
        if not re.fullmatch(r"[0-9a-f]{32}", trace_id):
            raise NotFoundError("trace")
        key = f"insights:trace:{trace_id}"
        cached = await self.redis.get(key)
        if cached:
            detail = TraceDetail.model_validate_json(cached)
        else:
            obs = await self._call(
                self.reader.observations(
                    traceId=trace_id,
                    limit=200,
                    fields="core,basic,time,io,metadata,model,usage,metrics,trace_context",
                )
            )
            root = next((o for o in obs if o.get("isRootObservation")), None)
            if root is None:
                return None
            if root.get("userId") != str(owner_id):
                raise NotFoundError("trace")
            detail = await self._detail(trace_id, root, obs)
            if all(o.get("endTime") for o in obs):
                await self.redis.set(key, detail.model_dump_json(), ex=TRACE_TTL_S)
        if detail.user_id != str(owner_id):
            raise NotFoundError("trace")
        # Оценку не кэшируем вместе с трейсом: пользователь может передумать
        fb = (await self._feedback_by_trace(owner_id, [trace_id])).get(trace_id)
        return detail.model_copy(update={"feedback": None if fb is None else fb > 0})

    async def _detail(
        self,
        trace_id: str,
        root: dict[str, Any],
        obs: list[dict[str, Any]],
    ) -> TraceDetail:
        t0 = _dt(root["startTime"])
        by_id = {o["id"]: o for o in obs}

        def depth(o: dict[str, Any]) -> int:
            d, cur = 0, o
            while cur is not root and cur.get("parentObservationId") in by_id:
                cur = by_id[cur["parentObservationId"]]
                d += 1
            return d

        def ms(start: str, end: str | None) -> float:
            return ((_dt(end) if end else _dt(start)) - _dt(start)).total_seconds() * 1000

        spans: list[SpanRow] = []
        prompt: list[dict[str, str]] = []
        for o in sorted(obs, key=lambda o: (o["startTime"], depth(o))):
            details, hits = self._span_details(o)
            usage = o.get("usageDetails") or None
            if o.get("type") == "GENERATION":
                messages = _json(o.get("input"))
                if isinstance(messages, list):
                    prompt = [
                        {"role": str(m.get("role")), "content": str(m.get("content") or "")}
                        for m in messages
                        if isinstance(m, dict)
                    ]
            spans.append(
                SpanRow(
                    id=o["id"],
                    parent_id=o.get("parentObservationId"),
                    name=o["name"],
                    type=o.get("type") or "SPAN",
                    depth=depth(o),
                    offset_ms=(_dt(o["startTime"]) - t0).total_seconds() * 1000,
                    duration_ms=ms(o["startTime"], o.get("endTime")),
                    level=o.get("level") or "DEFAULT",
                    status_message=o.get("statusMessage") or None,
                    model=o.get("model") or None,
                    usage={k: int(v) for k, v in usage.items()} if usage else None,
                    cost_usd=_f(o.get("totalCost")),
                    ttft_ms=(_f(o.get("timeToFirstToken")) or 0) * 1000 or None,
                    details=details,
                    hits=hits,
                )
            )
        answer = _json(root.get("output"))
        project = await self.project_url()
        return TraceDetail(
            trace_id=trace_id,
            name=root.get("traceName") or root["name"],
            user_id=root.get("userId") or "",
            started_at=t0,
            duration_ms=max((s.offset_ms + s.duration_ms for s in spans), default=0),
            agent=_agent_tag(root.get("tags")),
            session_id=root.get("sessionId") or None,
            title=self._title(root),
            answer=answer if isinstance(answer, str) else None,
            level=root.get("level") or "DEFAULT",
            status_message=root.get("statusMessage") or None,
            spans=spans,
            prompt=prompt,
            cost_usd=sum(s.cost_usd or 0 for s in spans),
            feedback=None,
            langfuse_url=f"{project}/traces/{trace_id}" if project else None,
        )

    @staticmethod
    def _span_details(o: dict[str, Any]) -> tuple[dict[str, str], list[SearchHit]]:
        merged: dict[str, Any] = {}
        hits: list[SearchHit] = []
        for part in (o.get("modelParameters"), o.get("metadata"), _json(o.get("input"))):
            if isinstance(part, dict):
                merged.update(part)
        out = _json(o.get("output"))
        if isinstance(out, dict):
            merged.update(out)
        elif isinstance(out, list) and o.get("type") == "RETRIEVER":
            hits = [
                SearchHit(
                    score=_f(h.get("score")) or 0.0,
                    chunk_id=str(h.get("chunk_id")),
                    label=" · ".join(str(x) for x in (h.get("book_title"), h.get("chapter")) if x)
                    or "—",
                )
                for h in out
                if isinstance(h, dict)
            ]
        details = {k: _short(merged[k], 80) for k in _DETAIL_KEYS if merged.get(k) is not None}
        return details, hits
