"""InsightsService на записанных ответах Langfuse API: без сети, БД и Redis."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar
from uuid import UUID, uuid4

import pytest

from rag_agents.core.config import Settings
from rag_agents.core.langfuse_api import LangfuseRateLimitedError, LangfuseReader
from rag_agents.domain.answers import AnswerUsage
from rag_agents.domain.insights import AnswerFact, IngestFact, Period
from rag_agents.llm.prices import PriceTable
from rag_agents.services.errors import NotFoundError
from rag_agents.services.insights import (
    InsightsDisabledError,
    InsightsService,
    InsightsUnavailableError,
)
from rag_agents.services.insights_stats import build_overview, percentile
from rag_agents.web.templating import templates

OWNER = uuid4()
AGENT_ID = uuid4()
TRACE = "a" * 32
PRICES = PriceTable.load(Path("configs/llm_prices.yaml"))


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.data.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.data[key] = value


class FakeReader:
    """Отвечает как Langfuse API; запоминает запросы, чтобы проверить фильтр по владельцу."""

    enabled = True
    base_url = "https://cloud.langfuse.com"

    def __init__(self, trace_owner: UUID = OWNER, rate_limited: bool = False) -> None:
        self.rate_limited = rate_limited
        self.calls = 0
        self.observation_calls: list[dict[str, Any]] = []
        self.trace_owner = trace_owner

    async def project_id(self) -> str:
        return "proj1"

    async def observations(self, **params: Any) -> list[dict[str, Any]]:
        self.calls += 1
        if self.rate_limited:
            raise LangfuseRateLimitedError("/v2/observations", 3600)
        self.observation_calls.append(params)
        if params.get("traceId"):
            return self._trace()
        if params.get("type") == "GENERATION":
            assert "metrics" in params["fields"]  # timeToFirstToken живёт в группе metrics
            return [{"traceId": TRACE, "totalCost": 0.0016, "timeToFirstToken": 1.6,
                     "usageDetails": {"input": 2000, "output": 700, "total": 2700}}]  # fmt: skip
        return [
            {"traceId": TRACE, "name": "query", "traceName": "query",
             "startTime": "2026-09-27T18:45:00.000Z", "latency": 4.9, "level": "DEFAULT",
             "tags": ["Толстой"], "sessionId": "chat-1", "input": json.dumps("В чём смысл жизни?")},
            {"traceId": "b" * 32, "name": "ingest", "traceName": "ingest",
             "startTime": "2026-09-27T18:40:00.000Z", "latency": 1.0, "level": "ERROR",
             "statusMessage": "PermanentError: empty", "tags": ["Толстой", "ingest"],
             "input": json.dumps({"filename": "пустой.txt"})},
        ]  # fmt: skip

    def _trace(self) -> list[dict[str, Any]]:
        sdk_meta = {"scope.attributes.public_key": "pk-lf-secret-ish", "agent_id": "x"}
        return [
            {"id": "r", "traceId": TRACE, "isRootObservation": True, "parentObservationId": "ext",
             "name": "query", "traceName": "query", "type": "SPAN", "userId": str(self.trace_owner),
             "startTime": "2026-09-27T18:45:00.000Z", "endTime": "2026-09-27T18:45:05.000Z",
             "tags": ["Толстой"], "sessionId": "chat-1", "metadata": sdk_meta,
             "input": json.dumps("Вопрос?"), "output": json.dumps("Ответ [1].")},
            {"id": "s", "traceId": TRACE, "parentObservationId": "r", "name": "qdrant_search",
             "type": "RETRIEVER", "startTime": "2026-09-27T18:45:00.200Z",
             "endTime": "2026-09-27T18:45:00.300Z", "metadata": sdk_meta,
             "input": json.dumps({"top_k": 6, "agent_id": "x"}),
             "output": json.dumps([{"chunk_id": "c1", "score": 0.61, "book_title": "Исповедь"}])},
            {"id": "g", "traceId": TRACE, "parentObservationId": "r", "name": "llm_generate",
             "type": "GENERATION", "startTime": "2026-09-27T18:45:00.300Z",
             "endTime": "2026-09-27T18:45:05.000Z", "model": "deepseek-flash",
             "modelParameters": {"reasoning_effort": "low"}, "totalCost": 0.0016,
             "timeToFirstToken": 1.6, "usageDetails": {"input": 2000, "output": 700},
             "input": json.dumps([{"role": "system", "content": "Правила"},
                                  {"role": "user", "content": "Вопрос?"}])},
        ]  # fmt: skip


class Service(InsightsService):
    """PG подменён: факты и оценки — из памяти."""

    answers: ClassVar[list[AnswerFact]] = []
    docs: ClassVar[list[IngestFact]] = []

    async def _facts(
        self, owner_id: UUID, since: datetime
    ) -> tuple[list[AnswerFact], list[IngestFact]]:
        return (self.answers, self.docs) if owner_id == OWNER else ([], [])

    async def _feedback_by_trace(self, owner_id: UUID, trace_ids: list[str]) -> dict[str, int]:
        return {TRACE: 1} if owner_id == OWNER else {}


def service(reader: FakeReader | None = None) -> Service:
    return Service(reader or FakeReader(), FakeRedis(), None, PRICES)  # type: ignore[arg-type]


NOW = datetime(2026, 9, 28, 12, 30, tzinfo=UTC)  # пн, off-peak


def usage(total: int, ttft: int | None, *, cost: float | None = None, **kw: Any) -> AnswerUsage:
    return AnswerUsage(
        provider=kw.get("provider", "deepseek"),
        model=kw.get("model", "deepseek-flash"),
        reasoning_effort="low",
        input_tokens=kw.get("input_tokens", 2000),
        cached_input_tokens=kw.get("cached", 0),
        output_tokens=kw.get("output_tokens", 500),
        t_embed_ms=kw.get("embed", 150),
        t_search_ms=kw.get("search", 40),
        t_retrieval_ms=kw.get("retrieval", 200),
        t_first_token_ms=ttft,
        t_total_ms=total,
        cost_usd=cost,
    )


def answer(
    minutes_ago: int, u: AnswerUsage | None, *, agent: str = "Толстой", **kw: Any
) -> AnswerFact:
    return AnswerFact(
        message_id=uuid4(),
        chat_id=kw.get("chat_id", uuid4()),
        agent_id=AGENT_ID if agent == "Толстой" else OTHER_AGENT,
        agent_name=agent,
        created_at=NOW - timedelta(minutes=minutes_ago),
        status=kw.get("status", "done"),
        question=kw.get("question", "Вопрос?"),
        usage=u,
        refused=False,
        feedback=kw.get("feedback"),
        trace_id=kw.get("trace_id"),
    )


OTHER_AGENT = uuid4()
ANSWERS = [
    answer(5, usage(4000, 1000, cost=0.002), feedback=1, trace_id=TRACE, question="Смысл?"),
    answer(10, usage(6000, 2000, cost=0.004), feedback=-1),
    # старый ответ без cost_usd — пересчитывается по таблице цен (off-peak)
    answer(70, usage(8000, 3000, input_tokens=1_000_000, output_tokens=0, cached=0)),
    answer(80, usage(1000, None, provider="none", model="none"), agent="Другой"),
    answer(90, None, status="error"),
]
DOCS = [
    IngestFact(
        document_id=uuid4(), agent_id=AGENT_ID, agent_name="Толстой", filename="book.txt",
        status="done", created_at=NOW - timedelta(minutes=30),
        started_at=NOW - timedelta(minutes=30), finished_at=NOW - timedelta(minutes=29),
        chunks_total=161, error_message=None,
        timings={"parse_ms": 100, "chunk_ms": 900, "save_chunks_ms": 300, "embed_ms": 30000,
                 "upsert_ms": 800},
    ),
    IngestFact(
        document_id=uuid4(), agent_id=AGENT_ID, agent_name="Толстой", filename="пустой.txt",
        status="failed", created_at=NOW - timedelta(minutes=31), started_at=None,
        finished_at=None, chunks_total=None, error_message="Файл пустой",
    ),
]  # fmt: skip


def test_percentile_is_nearest_rank() -> None:
    assert percentile([], 50) is None
    assert percentile([5], 95) == 5
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile(range(1, 101), 95) == 95


def test_overview_from_pg_facts() -> None:
    o = build_overview(Period.DAY, NOW, ANSWERS, DOCS, PRICES, "https://lf/project/p")
    k = o.kpi
    assert (k.questions, k.llm_calls, k.ingests, k.errors) == (5, 3, 2, 2)
    # 0.002 + 0.004 + 1M входных токенов off-peak (0.30 / 2) + отказ без LLM (0)
    assert k.cost_usd == pytest.approx(0.006 + 0.15)
    assert k.cost_per_question == pytest.approx(0.156 / 3)
    assert (k.feedback_up, k.feedback_down) == (1, 1)
    assert k.ttft_p50_ms == 2000
    assert k.latency_p95_ms == 8000

    assert len(o.series) == 25  # 24 часа + текущий
    assert sum(p.questions for p in o.series) == 5

    tolstoy, other = o.agents
    assert (tolstoy.name, tolstoy.agent_id, tolstoy.questions) == ("Толстой", AGENT_ID, 4)
    assert tolstoy.feedback_up_rate == 0.5
    assert (other.name, other.cost_usd) == ("Другой", 0.0)

    stages = {(s.trace_name, s.name): s for s in o.stages}
    assert stages[("query", "embed_query")].p50_ms == 150
    assert stages[("query", "llm: до первого токена")].p50_ms == 1800  # 2000 − 200 поиск
    assert stages[("query", "llm: генерация ответа")].count == 3
    assert stages[("ingest", "embed")].p50_ms == 30000
    assert [m.model for m in o.models] == ["deepseek/deepseek-flash"]

    first = o.recent[0]
    assert (first.title, first.trace_id, first.feedback, first.cost_usd) == (
        "Смысл?", TRACE, True, 0.002,
    )  # fmt: skip
    failed = next(r for r in o.recent if r.name == "ingest" and r.level == "ERROR")
    assert failed.status_message == "Файл пустой"
    assert failed.session_id is not None
    assert failed.session_id.startswith("document-")
    assert o.langfuse_project_url == "https://lf/project/p"


async def test_overview_uses_only_own_facts_and_no_langfuse_metrics() -> None:
    reader = FakeReader()
    Service.answers, Service.docs = ANSWERS, DOCS
    svc = service(reader)
    own = await svc.overview(OWNER, Period.WEEK)
    assert own.kpi.questions == 5
    assert reader.calls == 0  # сводка не ходит в Langfuse (кроме id проекта для ссылки)
    foreign = await svc.overview(uuid4(), Period.WEEK)
    assert foreign.kpi.questions == 0


async def test_rate_limit_blocks_further_calls_until_reset() -> None:
    reader = FakeReader(rate_limited=True)
    svc = service(reader)
    with pytest.raises(InsightsUnavailableError, match="Лимит API Langfuse"):
        await svc.trace(OWNER, TRACE)
    assert reader.calls == 1
    with pytest.raises(InsightsUnavailableError, match="Лимит API Langfuse"):
        await svc.trace(OWNER, TRACE)
    assert reader.calls == 1  # второй раз в Langfuse не пошли


async def test_trace_detail_tree_hits_prompt_and_no_sdk_metadata() -> None:
    t = await service().trace(OWNER, TRACE)
    assert t is not None
    assert [s.name for s in t.spans] == ["query", "qdrant_search", "llm_generate"]
    assert [s.depth for s in t.spans] == [0, 1, 1]
    assert t.duration_ms == pytest.approx(5000)
    search = t.spans[1]
    assert search.offset_ms == pytest.approx(200)
    assert search.hits[0].score == 0.61
    assert search.details["top_k"] == "6"
    gen = t.spans[2]
    assert gen.ttft_ms == pytest.approx(1600)
    assert gen.details["reasoning_effort"] == "low"
    assert [m["role"] for m in t.prompt] == ["system", "user"]
    assert (t.answer, t.feedback, t.cost_usd) == ("Ответ [1].", True, 0.0016)
    assert t.langfuse_url == f"https://cloud.langfuse.com/project/proj1/traces/{TRACE}"
    dumped = t.model_dump_json()
    assert "pk-lf" not in dumped
    assert "agent_id" not in dumped  # служебные и внутренние ключи наружу не уходят


async def test_foreign_trace_is_not_found_even_from_cache() -> None:
    svc = service(FakeReader(trace_owner=uuid4()))
    with pytest.raises(NotFoundError):
        await svc.trace(OWNER, TRACE)

    own = service()
    await own.trace(OWNER, TRACE)  # кладёт в кэш
    with pytest.raises(NotFoundError):
        await own.trace(uuid4(), TRACE)


async def test_invalid_trace_id_is_not_found() -> None:
    with pytest.raises(NotFoundError):
        await service().trace(OWNER, "../../etc")


async def test_disabled_without_keys() -> None:
    reader = LangfuseReader(Settings(_env_file=None, app_env="dev"))
    svc = InsightsService(reader, FakeRedis(), None, PRICES)  # type: ignore[arg-type]
    assert not svc.enabled
    assert await svc.project_url() is None
    with pytest.raises(InsightsDisabledError):
        await svc.trace(OWNER, TRACE)
    await reader.aclose()


async def test_templates_render_overview_and_trace() -> None:
    svc = service()
    overview = build_overview(Period.DAY, NOW, ANSWERS, DOCS, PRICES, None)
    trace = await svc.trace(OWNER, TRACE)
    templates.env.globals["insights_enabled"] = True
    html = templates.get_template("fragments/insights_overview.html").render(o=overview)
    assert "Толстой" in html
    assert "series-data" in html
    assert "до первого токена" in html
    html = templates.get_template("fragments/insights_trace.html").render(t=trace, trace_id=TRACE)
    assert "qdrant_search" in html
    assert "0.610" in html
    tpl = templates.get_template("fragments/insights_trace.html")
    pending = tpl.render(t=None, trace_id=TRACE, attempt=2, give_up=False)
    assert 'hx-trigger="load delay:3s"' in pending
    assert "attempt=3" in pending
    gone = tpl.render(t=None, trace_id=TRACE, attempt=10, give_up=True)
    assert "hx-get" not in gone
    assert "не найден" in gone
