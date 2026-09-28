"""Тестовые двойники, общие для unit и integration."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Literal, Self, Unpack

from rag_agents.core.observability import ObservationFields, ObservationType, trace_id_for
from rag_agents.domain.tasks import IngestDocumentTask
from rag_agents.llm.base import LLMChunk, LLMRequest, LLMUsage

VECTOR = [0.5, 0.5, 0.5, 0.5]


class FakeEmbedder:
    model = "fake-embed"
    dim = len(VECTOR)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [VECTOR for _ in texts]


class FakeLLM:
    name = "deepseek"
    model = "deepseek-flash"
    reasoning_effort: str | None = "low"

    async def stream(self, req: LLMRequest) -> AsyncIterator[LLMChunk]:
        for delta in ("Смысл ", "жизни [1]."):
            yield LLMChunk(delta=delta)
        usage = LLMUsage(input_tokens=100, cached_input_tokens=40, output_tokens=30)
        yield LLMChunk(finish_reason="stop", usage=usage)


@dataclass
class RecordingPublisher:
    """TaskPublisher без RabbitMQ: запоминает опубликованные задачи."""

    tasks: list[IngestDocumentTask] = field(default_factory=list)

    def publish_ingest(self, task: IngestDocumentTask) -> None:
        self.tasks.append(task)


@dataclass
class RecordedSpan:
    name: str
    as_type: str
    trace_id: str
    parent: "RecordedSpan | None"
    fields: dict[str, Any]
    trace_attrs: dict[str, Any] = field(default_factory=dict)
    ended: int = 0
    children: list["RecordedSpan"] = field(default_factory=list)
    recorder: "RecordingTracer | None" = None

    def child(
        self, name: str, *, as_type: ObservationType = "span", **fields: Unpack[ObservationFields]
    ) -> "RecordedSpan":
        span = RecordedSpan(
            name, as_type, self.trace_id, self, dict(fields), recorder=self.recorder
        )
        self.children.append(span)
        assert self.recorder is not None
        self.recorder.spans.append(span)
        return span

    def update(self, **fields: Unpack[ObservationFields]) -> None:
        self.fields.update(fields)

    def end(self, **fields: Unpack[ObservationFields]) -> None:
        # Контракт как у LangfuseSpan: повторный end игнорируется (finally после except)
        if self.ended:
            return
        self.fields.update(fields)
        self.ended += 1

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.end(**({"level": "ERROR"} if exc else {}))  # type: ignore[typeddict-item]


@dataclass
class RecordingTracer:
    """Tracer, который пишет span-ы и score-ы в память: проверяем форму трейса без сети."""

    enabled: bool = True
    spans: list[RecordedSpan] = field(default_factory=list)
    scores: dict[str, dict[str, Any]] = field(default_factory=dict)
    flushed: int = 0

    def trace_id_for(self, seed: str) -> str:
        return trace_id_for(seed)

    def start_trace(
        self,
        name: str,
        *,
        trace_id: str,
        user_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        **fields: Unpack[ObservationFields],
    ) -> RecordedSpan:
        span = RecordedSpan(
            name,
            "span",
            trace_id,
            None,
            dict(fields),
            trace_attrs={"user_id": user_id, "session_id": session_id, "tags": tags},
            recorder=self,
        )
        self.spans.append(span)
        return span

    def score(
        self,
        *,
        trace_id: str,
        name: str,
        value: float,
        score_id: str,
        data_type: Literal["NUMERIC", "BOOLEAN"] = "NUMERIC",
        comment: str | None = None,
    ) -> None:
        # Как в Langfuse: score с тем же id перезаписывается
        self.scores[score_id] = {"trace_id": trace_id, "name": name, "value": value}

    def flush(self) -> None:
        self.flushed += 1

    def shutdown(self) -> None:
        self.flush()

    def roots(self) -> list[RecordedSpan]:
        return [s for s in self.spans if s.parent is None]
