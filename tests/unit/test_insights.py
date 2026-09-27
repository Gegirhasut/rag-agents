"""InsightsService на записанных ответах Langfuse API: без сети, БД и Redis."""

import json
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from rag_agents.core.config import Settings
from rag_agents.core.langfuse_api import LangfuseReader
from rag_agents.domain.chats import FeedbackStat
from rag_agents.domain.insights import Period
from rag_agents.services.errors import NotFoundError
from rag_agents.services.insights import InsightsDisabledError, InsightsService
from rag_agents.web.templating import templates

OWNER = uuid4()
AGENT_ID = uuid4()
TRACE = "a" * 32


# Ответы Metrics API v2 по виду запроса (числа — то строкой, то числом, как в реальном API)
METRICS: dict[str, list[dict[str, Any]]] = {
    "series": [
        {"time_dimension": "2026-09-27T18:00:00Z", "count_count": "3", "sum_totalCost": 0.004,
         "p50_timeToFirstToken": 1500},
        {"time_dimension": "2026-09-27T19:00:00Z", "count_count": "0", "sum_totalCost": None,
         "p50_timeToFirstToken": None},
    ],
    "errors": [{"traceName": "ingest", "count_count": "1"}],
    "traceName": [
        {"traceName": "query", "count_count": "4", "p50_latency": 5000, "p95_latency": 9000},
        {"traceName": "ingest", "count_count": "2", "p50_latency": 20000,
         "p95_latency": 30000},
    ],
    "tags": [
        {"tags": ["Толстой"], "count_count": "4", "sum_totalCost": 0.006,
         "sum_totalTokens": 9000, "p50_timeToFirstToken": 1400,
         "p50_latency": 5000, "p95_latency": 9000},
        {"tags": ["Удалённый", "ingest"], "count_count": "1", "sum_totalCost": 0,
         "sum_totalTokens": 0, "p50_latency": 100, "p95_latency": 100},
    ],
    "traceName-name": [
        {"traceName": "query", "name": "llm_generate", "count_count": "4",
         "p50_latency": 4500, "p95_latency": 8000},
        {"traceName": "query", "name": "embed_query", "count_count": "4",
         "p50_latency": 150, "p95_latency": 300},
        {"traceName": "ingest", "name": "embed_batch 1/6", "count_count": "1",
         "p50_latency": 20000, "p95_latency": 20000},
        {"traceName": "connectivity-check", "name": "ping", "count_count": "1",
         "p50_latency": 1, "p95_latency": 1},
    ],
    "providedModelName": [
        {"providedModelName": "deepseek-flash", "count_count": "4", "sum_totalCost": 0.006,
         "sum_inputTokens": "8000", "sum_outputTokens": 1000},
    ],
    "totals": [
        {"count_count": "4", "sum_totalCost": 0.006, "sum_inputTokens": "8000",
         "sum_outputTokens": 1000, "p50_timeToFirstToken": 1400, "p95_timeToFirstToken": 2600},
    ],
}  # fmt: skip


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

    def __init__(self, trace_owner: UUID = OWNER) -> None:
        self.metric_queries: list[dict[str, Any]] = []
        self.observation_calls: list[dict[str, Any]] = []
        self.trace_owner = trace_owner

    async def project_id(self) -> str:
        return "proj1"

    async def metrics(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        self.metric_queries.append(query)
        dims = tuple(d["field"] for d in query["dimensions"])
        filters = {f["column"]: f["value"] for f in query["filters"]}
        if "timeDimension" in query:
            key = "series"
        elif dims == ("traceName",) and filters.get("level") == "ERROR":
            key = "errors"
        else:
            key = "-".join(dims) or "totals"
        return METRICS[key]

    async def observations(self, **params: Any) -> list[dict[str, Any]]:
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
    """PG подменён: агенты и оценки берутся из памяти."""

    async def _agent_ids(self, owner_id: UUID) -> dict[str, UUID]:
        return {"Толстой": AGENT_ID}

    async def _feedback_stats(self, owner_id: UUID, since: datetime) -> list[FeedbackStat]:
        return [FeedbackStat(agent_id=AGENT_ID, up=3, down=1)]

    async def _feedback_by_trace(self, owner_id: UUID, trace_ids: list[str]) -> dict[str, int]:
        return {TRACE: 1} if owner_id == OWNER else {}


def service(reader: FakeReader | None = None) -> Service:
    return Service(reader or FakeReader(), FakeRedis(), None)  # type: ignore[arg-type]


async def test_overview_filters_every_query_by_owner_and_parses_numbers() -> None:
    reader = FakeReader()
    o = await service(reader).overview(OWNER, Period.DAY)

    owner_filter = {"column": "userId", "operator": "=", "value": str(OWNER), "type": "string"}
    assert reader.metric_queries
    assert all(owner_filter in q["filters"] for q in reader.metric_queries)
    assert all(c.get("userId") == str(OWNER) for c in reader.observation_calls)

    k = o.kpi
    assert (k.questions, k.ingests, k.errors, k.llm_calls) == (4, 2, 1, 4)
    assert k.cost_usd == pytest.approx(0.006)
    assert k.cost_per_question == pytest.approx(0.0015)
    assert (k.feedback_up, k.feedback_down) == (3, 1)
    assert o.series[0].questions == 3
    assert o.series[1].cost_usd == 0.0

    [tolstoy, deleted] = o.agents
    assert (tolstoy.name, tolstoy.agent_id, tolstoy.questions) == ("Толстой", AGENT_ID, 4)
    assert tolstoy.feedback_up_rate == 0.75
    assert (deleted.name, deleted.agent_id) == ("Удалённый", None)  # тег "ingest" — не агент

    names = [(s.trace_name, s.name) for s in o.stages]
    assert names == [("query", "llm_generate"), ("query", "embed_query")]
    assert o.models[0].model == "deepseek-flash"
    assert o.langfuse_project_url == "https://cloud.langfuse.com/project/proj1"

    query_row, ingest_row = o.recent
    assert query_row.title == "В чём смысл жизни?"
    assert (query_row.cost_usd, query_row.tokens, query_row.feedback) == (0.0016, 2700, True)
    assert query_row.ttft_ms == pytest.approx(1600)
    assert (ingest_row.title, ingest_row.level) == ("пустой.txt", "ERROR")


async def test_overview_is_cached() -> None:
    reader = FakeReader()
    svc = service(reader)
    await svc.overview(OWNER, Period.WEEK)
    n = len(reader.metric_queries)
    await svc.overview(OWNER, Period.WEEK)
    assert len(reader.metric_queries) == n


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
    svc = InsightsService(reader, FakeRedis(), None)  # type: ignore[arg-type]
    assert not svc.enabled
    assert await svc.project_url() is None
    with pytest.raises(InsightsDisabledError):
        await svc.overview(OWNER, Period.DAY)
    await reader.aclose()


async def test_templates_render_overview_and_trace() -> None:
    svc = service()
    overview = await svc.overview(OWNER, Period.DAY)
    trace = await svc.trace(OWNER, TRACE)
    templates.env.globals["insights_enabled"] = True
    html = templates.get_template("fragments/insights_overview.html").render(o=overview)
    assert "Толстой" in html
    assert "series-data" in html
    assert "llm_generate" in html
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
