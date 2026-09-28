"""Контракты страницы «Аналитика»: агрегаты и трейсы из Langfuse в нашем виде."""

from datetime import datetime, timedelta
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel

from rag_agents.domain.answers import AnswerUsage


class Period(StrEnum):
    DAY = "24h"
    WEEK = "7d"
    MONTH = "30d"

    @property
    def delta(self) -> timedelta:
        return {"24h": timedelta(days=1), "7d": timedelta(days=7), "30d": timedelta(days=30)}[
            self.value
        ]

    @property
    def granularity(self) -> str:
        return "hour" if self is Period.DAY else "day"

    @property
    def label(self) -> str:
        return {"24h": "24 часа", "7d": "7 дней", "30d": "30 дней"}[self.value]


class Kpi(BaseModel):
    questions: int = 0
    llm_calls: int = 0
    ingests: int = 0
    errors: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    ttft_p50_ms: float | None = None
    ttft_p95_ms: float | None = None
    feedback_count: int = 0
    feedback_up_rate: float | None = None  # доля 👍 среди оценок, 0..1

    @property
    def cost_per_question(self) -> float | None:
        return self.cost_usd / self.llm_calls if self.llm_calls else None

    @property
    def feedback_up(self) -> int:
        return round(self.feedback_count * (self.feedback_up_rate or 0))

    @property
    def feedback_down(self) -> int:
        return self.feedback_count - self.feedback_up


class TimePoint(BaseModel):
    ts: datetime
    questions: int
    cost_usd: float
    ttft_p50_ms: float | None


class AgentStats(BaseModel):
    name: str
    agent_id: UUID | None  # None — агент удалён или переименован
    questions: int = 0
    cost_usd: float = 0.0
    tokens: int = 0
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    ttft_p50_ms: float | None = None
    feedback_count: int = 0
    feedback_up_rate: float | None = None


class StageStats(BaseModel):
    """Время по шагам пайплайна: где тратятся миллисекунды."""

    trace_name: str
    name: str
    count: int
    p50_ms: float | None
    p95_ms: float | None


class ModelStats(BaseModel):
    model: str
    calls: int
    cost_usd: float
    input_tokens: int
    output_tokens: int


class TraceRow(BaseModel):
    trace_id: str | None  # None — трейсинг был выключен
    name: str
    started_at: datetime
    latency_ms: float | None
    level: str
    status_message: str | None
    agent: str | None
    session_id: str | None
    title: str  # вопрос или имя файла
    cost_usd: float | None = None
    tokens: int | None = None
    ttft_ms: float | None = None
    feedback: bool | None = None  # True = 👍, False = 👎


class SearchHit(BaseModel):
    score: float
    chunk_id: str
    label: str


class SpanRow(BaseModel):
    id: str
    parent_id: str | None
    name: str
    type: str
    depth: int
    offset_ms: float
    duration_ms: float
    level: str
    status_message: str | None
    model: str | None = None
    usage: dict[str, int] | None = None
    cost_usd: float | None = None
    ttft_ms: float | None = None
    details: dict[str, str] = {}  # отобранные поля input/output/metadata (без служебных SDK)
    hits: list[SearchHit] = []  # для retriever: найденные чанки со score

    @property
    def is_error(self) -> bool:
        return self.level == "ERROR"


class TraceDetail(BaseModel):
    trace_id: str
    name: str
    user_id: str  # владелец: проверяется при чтении из кэша
    started_at: datetime
    duration_ms: float
    agent: str | None
    session_id: str | None
    title: str
    answer: str | None
    level: str
    status_message: str | None
    spans: list[SpanRow]
    prompt: list[dict[str, str]] = []  # messages промпта LLM (role, content)
    cost_usd: float
    feedback: bool | None
    langfuse_url: str | None


class InsightsOverview(BaseModel):
    period: Period
    generated_at: datetime
    kpi: Kpi
    series: list[TimePoint]
    agents: list[AgentStats]
    stages: list[StageStats]
    models: list[ModelStats]
    recent: list[TraceRow]
    langfuse_project_url: str | None


class AnswerFact(BaseModel):
    """Ответ ассистента из PG — сырьё для сводки «Аналитики»."""

    message_id: UUID
    chat_id: UUID
    agent_id: UUID
    agent_name: str
    created_at: datetime
    status: str
    question: str | None
    usage: AnswerUsage | None
    refused: bool | None
    feedback: int | None
    trace_id: str | None


class IngestFact(BaseModel):
    document_id: UUID
    agent_id: UUID
    agent_name: str
    filename: str
    status: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    chunks_total: int | None
    error_message: str | None
    timings: dict[str, int] = {}  # мс по шагам ingest (documents.meta.timings)

    @property
    def duration_ms(self) -> float | None:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds() * 1000
        return None
