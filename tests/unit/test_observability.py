from collections.abc import Iterator
from uuid import uuid4

import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rag_agents.core.config import Settings
from rag_agents.core.observability import LangfuseTracer, NoopTracer, build_tracer, trace_id_for


def settings(**kw: object) -> Settings:
    return Settings(_env_file=None, **kw)  # type: ignore[call-arg, arg-type]


def test_without_keys_tracer_is_noop() -> None:
    tracer = build_tracer(settings(app_env="dev"))
    assert isinstance(tracer, NoopTracer)
    span = tracer.start_trace("query", trace_id=tracer.trace_id_for("m"), user_id="u")
    with span.child("embed", as_type="embedding") as child:
        child.update(output={"dim": 3})
    span.end(output="ok")  # no-op API не падает и не требует сети
    tracer.score(trace_id=span.trace_id, name="x", value=1, score_id="s")
    tracer.shutdown()


def test_test_env_never_traces_even_with_keys() -> None:
    s = settings(app_env="test", langfuse_public_key="pk", langfuse_secret_key="sk")
    assert not s.langfuse_active
    assert isinstance(build_tracer(s), NoopTracer)


def test_keys_enable_tracing_outside_tests() -> None:
    s = settings(app_env="dev", langfuse_public_key="pk", langfuse_secret_key="sk")
    assert s.langfuse_active
    assert not settings(app_env="dev", langfuse_public_key="pk").langfuse_active


def test_base_url_falls_back_to_legacy_langfuse_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    monkeypatch.setenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
    assert settings().langfuse_base_url == "https://us.cloud.langfuse.com"
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://eu.example")
    assert settings().langfuse_base_url == "https://eu.example"  # новое имя главнее


def test_base_url_defaults_to_eu_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    monkeypatch.setenv("LANGFUSE_BASE_URL", "")
    assert settings().langfuse_base_url == "https://cloud.langfuse.com"


def test_trace_id_is_deterministic_langfuse_format() -> None:
    a, b = trace_id_for("query:1"), trace_id_for("query:1")
    assert a == b != trace_id_for("query:2")
    assert len(a) == 32
    int(a, 16)


@pytest.fixture
def exported() -> Iterator[tuple[LangfuseTracer, InMemorySpanExporter]]:
    exporter = InMemorySpanExporter()
    client = Langfuse(
        # SDK держит ресурсы-синглтоны по public_key: уникальный ключ = свежий клиент на тест
        public_key=f"pk-test-{uuid4().hex}",
        secret_key="sk-test",
        base_url="http://127.0.0.1:9",  # никуда не ходим: span-ы уходят в in-memory экспортёр
        span_exporter=exporter,
        tracer_provider=TracerProvider(),
    )
    tracer = LangfuseTracer(client)
    yield tracer, exporter
    client.shutdown()


def test_trace_attributes_reach_every_span(
    exported: tuple[LangfuseTracer, InMemorySpanExporter],
) -> None:
    tracer, exporter = exported
    tid = tracer.trace_id_for("query:msg")
    root = tracer.start_trace(
        "query", trace_id=tid, user_id="owner", session_id="chat", tags=["Толстой"], input="q"
    )
    with root.child("embed_query", as_type="embedding", model="bge-m3"):
        pass
    gen = root.child("llm_generate", as_type="generation", model="deepseek-flash")
    gen.end(output="a", usage_details={"input": 10, "input_cache_read": 2, "output": 5})
    root.end(output="a")
    tracer.flush()

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert set(spans) == {"query", "embed_query", "llm_generate"}
    for s in spans.values():
        assert s.context is not None
        assert f"{s.context.trace_id:032x}" == tid
        attrs = s.attributes or {}
        assert attrs["user.id"] == "owner"
        assert attrs["session.id"] == "chat"
        assert tuple(attrs["langfuse.trace.tags"]) == ("Толстой",)  # type: ignore[arg-type]
    assert spans["llm_generate"].parent is not None
    assert spans["llm_generate"].parent.span_id == spans["query"].context.span_id  # type: ignore[union-attr]
    assert (spans["llm_generate"].attributes or {})["langfuse.observation.type"] == "generation"


def test_exception_marks_span_as_error_and_end_is_idempotent(
    exported: tuple[LangfuseTracer, InMemorySpanExporter],
) -> None:
    tracer, exporter = exported
    root = tracer.start_trace("ingest", trace_id=tracer.trace_id_for("d"))
    with pytest.raises(RuntimeError), root.child("parse"):
        raise RuntimeError("broken file")
    root.end()
    root.end()  # finally + __exit__ не должны давать двойной end
    tracer.flush()
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert len(exporter.get_finished_spans()) == 2
    assert (spans["parse"].attributes or {})["langfuse.observation.level"] == "ERROR"


def test_sdk_failure_does_not_break_caller() -> None:
    class Broken:
        def start_observation(self, **_: object) -> None:
            raise RuntimeError("sdk down")

        def create_score(self, **_: object) -> None:
            raise RuntimeError("sdk down")

    tracer = LangfuseTracer(Broken())  # type: ignore[arg-type]
    span = tracer.start_trace("query", trace_id=tracer.trace_id_for("x"))
    span.child("embed").end()
    span.end(output="still works")
    tracer.score(trace_id=span.trace_id, name="user_feedback", value=1, score_id="s")
