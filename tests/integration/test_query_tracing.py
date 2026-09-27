"""Трейс вопроса и 👍/👎 → score: реальные PG и Qdrant; LLM, эмбеддер и tracer — фейки."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from qdrant_client import AsyncQdrantClient

from rag_agents.core.config import Settings
from rag_agents.core.db import Database
from rag_agents.domain.agents import AgentOut, AgentSettings
from rag_agents.domain.answers import DoneEvent, StreamEvent
from rag_agents.domain.documents import ChunkPayload
from rag_agents.llm.base import LLMChunk, LLMRequest, LLMUsage
from rag_agents.rag.index.qdrant import QdrantChunkIndex, chunk_point_id, collection_name
from rag_agents.repositories.agents import AgentRepository, UserRepository
from rag_agents.repositories.chats import ChatRepository
from rag_agents.services.errors import NotFoundError
from rag_agents.services.query import QueryService
from rag_agents.services.trace import TraceBus
from tests.fakes import RecordingTracer

pytestmark = pytest.mark.integration
DIM = 4
VECTOR = [0.5, 0.5, 0.5, 0.5]


class FakeEmbedder:
    model = "fake-embed"
    dim = DIM

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


async def _agent_with_chunk(
    db: Database, qdrant: AsyncQdrantClient, prefix: str, email: str
) -> AgentOut:
    collection = collection_name(prefix, "fake", DIM)
    async with db.uow() as uow:
        owner = await UserRepository(uow.session).get_or_create(email)
        agent = await AgentRepository(uow.session).create_with_index(
            owner,
            name="Толстой (тест)",
            slug=f"t-{uuid4().hex[:6]}",
            description="",
            persona_prompt=None,
            settings=AgentSettings(),
            embedding_model="fake",
            dim=DIM,
            collection=collection,
        )
        await uow.commit()
    index = QdrantChunkIndex(qdrant)
    await index.ensure_collection(collection, DIM)
    doc = uuid4()
    pid = chunk_point_id(doc, 1, 0)
    payload = ChunkPayload(
        agent_id=str(agent.id),
        document_id=str(doc),
        chunk_id=str(pid),
        ord=0,
        book_title="Исповедь",
        author="Л. Толстой",
        section_path=[],
        chapter_title=None,
        text="Жизнь есть бессмыслица…",
    )
    await index.upsert(collection, agent.id, [(pid, VECTOR, payload)])
    return agent


async def _collect(events: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [ev async for ev in events]


@pytest.fixture
def tracer() -> RecordingTracer:
    return RecordingTracer()


@pytest.fixture
def service(db: Database, qdrant: AsyncQdrantClient, tracer: RecordingTracer) -> QueryService:
    return QueryService(
        db,
        FakeEmbedder(),
        QdrantChunkIndex(qdrant),
        FakeLLM(),
        TraceBus(None, enabled=False),  # type: ignore[arg-type]  # Redis не нужен: шина выключена
        Settings(_env_file=None),  # type: ignore[call-arg]
        tracer,
    )


async def test_question_produces_trace_with_retrieval_and_generation(
    db: Database,
    qdrant: AsyncQdrantClient,
    collection_prefix: str,
    service: QueryService,
    tracer: RecordingTracer,
) -> None:
    agent = await _agent_with_chunk(db, qdrant, collection_prefix, f"tr-{uuid4().hex[:6]}@t")
    pair = await service.ask(agent.owner_id, agent.id, None, "В чём смысл жизни?")
    events = await _collect(await service.stream_answer(agent.owner_id, agent.id, pair.answer.id))
    assert isinstance(events[-1], DoneEvent)

    [root] = tracer.roots()
    assert root.name == "query"
    assert root.trace_id == tracer.trace_id_for(f"query:{pair.answer.id}")
    assert root.trace_attrs == {
        "user_id": str(agent.owner_id),
        "session_id": str(pair.answer.chat_id),
        "tags": ["Толстой (тест)"],
    }
    assert root.fields["output"] == "Смысл жизни [1]."
    assert root.ended == 1
    assert [c.name for c in root.children] == ["embed_query", "qdrant_search", "llm_generate"]

    embed, search, gen = root.children
    assert embed.as_type == "embedding"
    assert search.as_type == "retriever"
    assert search.fields["input"]["agent_id"] == str(agent.id)
    [hit] = search.fields["output"]
    assert set(hit) >= {"chunk_id", "document_id", "score"}

    assert gen.as_type == "generation"
    assert gen.fields["model"] == "deepseek-flash"
    assert gen.fields["model_parameters"]["reasoning_effort"] == "low"
    assert gen.fields["usage_details"] == {"input": 60, "input_cache_read": 40, "output": 30}
    assert "completion_start_time" in gen.fields
    assert gen.fields["input"][-1]["role"] == "user"
    assert all(s.ended == 1 for s in tracer.spans)

    async with db.session() as s:
        found = await ChatRepository(s).get_message(agent.id, agent.owner_id, pair.answer.id)
    assert found is not None
    assert found[0].trace_id == root.trace_id


async def test_feedback_is_saved_and_sent_as_score(
    db: Database,
    qdrant: AsyncQdrantClient,
    collection_prefix: str,
    service: QueryService,
    tracer: RecordingTracer,
) -> None:
    agent = await _agent_with_chunk(db, qdrant, collection_prefix, f"fb-{uuid4().hex[:6]}@t")
    pair = await service.ask(agent.owner_id, agent.id, None, "Вопрос?")
    await _collect(await service.stream_answer(agent.owner_id, agent.id, pair.answer.id))
    message_id: UUID = pair.answer.id

    msg = await service.feedback(agent.owner_id, agent.id, message_id, 1)
    assert msg.feedback == 1
    msg = await service.feedback(agent.owner_id, agent.id, message_id, -1)
    assert msg.feedback == -1

    # Один score на ответ: повторный клик перезаписывает, а не добавляет
    assert tracer.scores == {
        f"feedback-{message_id}": {
            "trace_id": tracer.trace_id_for(f"query:{message_id}"),
            "name": "user_feedback",
            "value": 0.0,
        }
    }

    with pytest.raises(NotFoundError):  # вопрос пользователя, а не ответ, оценивать нельзя
        await service.feedback(agent.owner_id, agent.id, pair.question.id, 1)


async def test_feedback_on_foreign_message_is_not_found(
    db: Database,
    qdrant: AsyncQdrantClient,
    collection_prefix: str,
    service: QueryService,
    tracer: RecordingTracer,
) -> None:
    alice = await _agent_with_chunk(db, qdrant, collection_prefix, f"a-{uuid4().hex[:6]}@t")
    bob = await _agent_with_chunk(db, qdrant, collection_prefix, f"b-{uuid4().hex[:6]}@t")
    pair = await service.ask(alice.owner_id, alice.id, None, "Вопрос?")
    await _collect(await service.stream_answer(alice.owner_id, alice.id, pair.answer.id))

    with pytest.raises(NotFoundError):
        await service.feedback(bob.owner_id, bob.id, pair.answer.id, 1)
    with pytest.raises(NotFoundError):  # свой агент, но чужое сообщение
        await service.feedback(bob.owner_id, alice.id, pair.answer.id, 1)
    assert tracer.scores == {}


async def test_feedback_stats_for_insights_are_scoped_to_owner(
    db: Database,
    qdrant: AsyncQdrantClient,
    collection_prefix: str,
    service: QueryService,
) -> None:
    """Оценки на странице «Аналитика» берутся из PG: только свои агенты и свои трейсы."""
    alice = await _agent_with_chunk(db, qdrant, collection_prefix, f"ia-{uuid4().hex[:6]}@t")
    bob = await _agent_with_chunk(db, qdrant, collection_prefix, f"ib-{uuid4().hex[:6]}@t")
    since = datetime.now(UTC) - timedelta(minutes=5)
    trace_ids = []
    for value in (1, 1, -1):
        pair = await service.ask(alice.owner_id, alice.id, None, "Вопрос?")
        await _collect(await service.stream_answer(alice.owner_id, alice.id, pair.answer.id))
        msg = await service.feedback(alice.owner_id, alice.id, pair.answer.id, value)  # type: ignore[arg-type]
        assert msg.trace_id is not None
        trace_ids.append(msg.trace_id)

    async with db.session() as s:
        repo = ChatRepository(s)
        [stat] = await repo.feedback_stats(alice.owner_id, since)
        assert (stat.agent_id, stat.up, stat.down) == (alice.id, 2, 1)
        assert await repo.feedback_stats(bob.owner_id, since) == []
        assert await repo.feedback_by_trace(alice.owner_id, trace_ids) == dict(
            zip(trace_ids, [1, 1, -1], strict=True)
        )
        assert await repo.feedback_by_trace(bob.owner_id, trace_ids) == {}
