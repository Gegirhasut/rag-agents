"""Трейсинг в Langfuse (ARCHITECTURE §14.2) за тонким протоколом Tracer.

Почему своя обёртка, а не @observe:
- ответ стримится из async-генератора, который крутится в отдельной задаче (with_heartbeat);
  неявный OTel-контекст через yield легко потерять, а явные span-объекты надёжнее;
- без ключей подставляется NoopTracer, и сервисы не проверяют «включено ли»;
- ошибка SDK не должна ронять ответ пользователю: всё обёрнуто в _safe (только warning в лог).

Атрибуты трейса (user_id, session_id, tags) в SDK v4 задаются только через контекстный
propagate_attributes. Поэтому каждый span создаётся внутри короткого with propagate_attributes:
атрибуты попадают на все span-ы трейса, и агрегаты Langfuse (стоимость по user или session)
считаются корректно.
"""

from collections.abc import Callable
from datetime import datetime
from types import TracebackType
from typing import Any, Literal, Protocol, Self, TypedDict, Unpack

import structlog
from langfuse import Langfuse, propagate_attributes

from rag_agents.core.config import Settings

log = structlog.get_logger()

ObservationType = Literal["span", "generation", "embedding", "retriever"]
Level = Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]


class ObservationFields(TypedDict, total=False):
    # Any: input/output/metadata уходят в SDK как произвольный JSON
    input: Any
    output: Any
    metadata: dict[str, Any]
    level: Level
    status_message: str
    model: str
    model_parameters: dict[str, str | int | float | bool | None]
    usage_details: dict[str, int]
    cost_details: dict[str, float]
    completion_start_time: datetime


class Span(Protocol):
    """Наблюдение в трейсе. Как контекстный менеджер завершается сам, при исключении — с ERROR."""

    @property
    def trace_id(self) -> str: ...

    def child(
        self, name: str, *, as_type: ObservationType = "span", **fields: Unpack[ObservationFields]
    ) -> "Span": ...

    def update(self, **fields: Unpack[ObservationFields]) -> None: ...

    def end(self, **fields: Unpack[ObservationFields]) -> None: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


class Tracer(Protocol):
    enabled: bool

    def trace_id_for(self, seed: str) -> str: ...

    def start_trace(
        self,
        name: str,
        *,
        trace_id: str,
        user_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        **fields: Unpack[ObservationFields],
    ) -> Span: ...

    def score(
        self,
        *,
        trace_id: str,
        name: str,
        value: float,
        score_id: str,
        data_type: Literal["NUMERIC", "BOOLEAN"] = "NUMERIC",
        comment: str | None = None,
    ) -> None: ...

    def flush(self) -> None: ...

    def shutdown(self) -> None: ...


def trace_id_for(seed: str) -> str:
    """Детерминированный trace_id (32 hex): по message_id или document_id трейс находится
    без записи в БД, а повторы (ретраи, повторный feedback) попадают в тот же трейс."""
    return Langfuse.create_trace_id(seed=seed)


def _safe(op: str, fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception as e:  # трейсинг не должен ломать основной сценарий
        log.warning("langfuse.error", op=op, error=repr(e)[:300])
        return None


class _BaseSpan:
    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc is None:
            self.end()
        else:
            self.end(level="ERROR", status_message=f"{type(exc).__name__}: {exc}"[:500])

    def end(self, **fields: Unpack[ObservationFields]) -> None:
        raise NotImplementedError


class NoopSpan(_BaseSpan):
    def __init__(self, trace_id: str) -> None:
        self._trace_id = trace_id

    @property
    def trace_id(self) -> str:
        return self._trace_id

    def child(
        self, name: str, *, as_type: ObservationType = "span", **fields: Unpack[ObservationFields]
    ) -> "NoopSpan":
        return self

    def update(self, **fields: Unpack[ObservationFields]) -> None:
        pass

    def end(self, **fields: Unpack[ObservationFields]) -> None:
        pass


class NoopTracer:
    enabled = False

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
    ) -> Span:
        return NoopSpan(trace_id)

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
        pass

    def flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


class _TraceAttrs(TypedDict, total=False):
    trace_name: str
    user_id: str
    session_id: str
    tags: list[str]


class LangfuseSpan(_BaseSpan):
    # Any: наблюдение SDK. start_observation перегружен по as_type (10 классов), а мы выбираем
    # тип в рантайме — строгая типизация на этой границе ничего не даёт
    def __init__(self, obs: Any, trace_id: str, attrs: _TraceAttrs) -> None:
        self._obs = obs
        self._trace_id = trace_id
        self._attrs = attrs
        self._ended = False

    @property
    def trace_id(self) -> str:
        return self._trace_id

    def child(
        self, name: str, *, as_type: ObservationType = "span", **fields: Unpack[ObservationFields]
    ) -> "LangfuseSpan":
        parent = self._obs

        def start() -> Any:
            if parent is None:
                return None
            with propagate_attributes(**self._attrs):
                return parent.start_observation(name=name, as_type=as_type, **fields)

        return LangfuseSpan(_safe("child", start), self._trace_id, self._attrs)

    def update(self, **fields: Unpack[ObservationFields]) -> None:
        if self._obs is not None and fields:
            obs = self._obs
            _safe("update", lambda: obs.update(**fields))

    def end(self, **fields: Unpack[ObservationFields]) -> None:
        # Идемпотентно: finally и __exit__ могут закрыть один и тот же span дважды
        if self._obs is None or self._ended:
            return
        self._ended = True
        self.update(**fields)
        obs = self._obs
        _safe("end", obs.end)


class LangfuseTracer:
    enabled = True

    def __init__(self, client: Langfuse) -> None:
        self._client = client

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
    ) -> Span:
        attrs: _TraceAttrs = {"trace_name": name}
        if user_id:
            attrs["user_id"] = user_id
        if session_id:
            attrs["session_id"] = session_id
        if tags:
            # Langfuse отбрасывает значения длиннее 200 символов
            attrs["tags"] = [t[:200] for t in tags]

        def start() -> Any:
            client: Any = self._client  # см. комментарий у LangfuseSpan
            with propagate_attributes(**attrs):
                return client.start_observation(
                    trace_context={"trace_id": trace_id}, name=name, **fields
                )

        return LangfuseSpan(_safe("start_trace", start), trace_id, attrs)

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
        _safe(
            "score",
            lambda: self._client.create_score(
                trace_id=trace_id,
                name=name,
                value=value,
                score_id=score_id,
                data_type=data_type,
                comment=comment,
            ),
        )

    def flush(self) -> None:
        _safe("flush", self._client.flush)

    def shutdown(self) -> None:
        _safe("shutdown", self._client.shutdown)


def build_tracer(settings: Settings, **client_kwargs: Any) -> Tracer:
    """Клиент Langfuse держит фоновый поток экспорта: в Celery его нужно создавать в дочернем
    процессе (worker_process_init), потоки через fork не переживают.

    client_kwargs (Any: параметры SDK) — для тестов: span_exporter, tracer_provider.
    """
    if not settings.langfuse_active or settings.langfuse_secret_key is None:
        return NoopTracer()
    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
        environment=settings.app_env,
        **client_kwargs,
    )
    log.info("langfuse.enabled", base_url=settings.langfuse_base_url)
    return LangfuseTracer(client)
