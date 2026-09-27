"""Изоляция данных агентов (SPEC NFR «Изоляция»): ни один путь чтения не видит чужое."""

import hashlib
from uuid import UUID, uuid4

import pytest
from qdrant_client import AsyncQdrantClient

from rag_agents.core.db import Database
from rag_agents.domain.agents import AgentOut, AgentSettings
from rag_agents.domain.documents import ChunkPayload
from rag_agents.domain.enums import MessageRole, MessageStatus
from rag_agents.rag.index.qdrant import QdrantChunkIndex, chunk_point_id, collection_name
from rag_agents.repositories.agents import AgentRepository, UserRepository
from rag_agents.repositories.chats import ChatRepository
from rag_agents.repositories.documents import DocumentRepository

pytestmark = pytest.mark.integration
DIM = 4


def payload(agent_id: UUID, document_id: UUID, pid: UUID, text: str) -> ChunkPayload:
    return ChunkPayload(
        agent_id=str(agent_id),
        document_id=str(document_id),
        chunk_id=str(pid),
        ord=0,
        book_title="Книга",
        author=None,
        section_path=[],
        chapter_title=None,
        text=text,
    )


async def test_vector_search_never_returns_other_agents_chunks(
    qdrant: AsyncQdrantClient, collection_prefix: str
) -> None:
    index = QdrantChunkIndex(qdrant)
    name = collection_name(collection_prefix, "bge-m3:567m", DIM)
    await index.ensure_collection(name, DIM)
    await index.ensure_collection(name, DIM)  # идемпотентно

    agent_a, agent_b = uuid4(), uuid4()
    doc_a, doc_b = uuid4(), uuid4()
    same_vector = [0.5, 0.5, 0.5, 0.5]  # одинаковые тексты у двух агентов
    for agent, doc in ((agent_a, doc_a), (agent_b, doc_b)):
        pts = []
        for i in range(3):
            pid = chunk_point_id(doc, 1, i)
            pts.append((pid, same_vector, payload(agent, doc, pid, f"общий текст {i}")))
        await index.upsert(name, agent, pts)

    hits_a = await index.search_dense(name, agent_a, same_vector, limit=10)
    hits_b = await index.search_dense(name, agent_b, same_vector, limit=10)

    assert len(hits_a) == 3
    assert {h.payload.agent_id for h in hits_a} == {str(agent_a)}
    assert {h.payload.agent_id for h in hits_b} == {str(agent_b)}

    # Удаление документа агента A не трогает агента B
    await index.delete_document(name, agent_a, doc_a)
    assert await index.search_dense(name, agent_a, same_vector, limit=10) == []
    assert len(await index.search_dense(name, agent_b, same_vector, limit=10)) == 3

    await qdrant.delete_collection(name)


async def test_system_page_reads_are_scoped_to_agent(
    qdrant: AsyncQdrantClient, collection_prefix: str
) -> None:
    """Карта векторов и «рентген» точки на /system не видят чужие точки."""
    index = QdrantChunkIndex(qdrant)
    name = collection_name(collection_prefix, "m", DIM)
    await index.ensure_collection(name, DIM)
    agent_a, agent_b, doc_a, doc_b = uuid4(), uuid4(), uuid4(), uuid4()
    pid_a, pid_b = chunk_point_id(doc_a, 1, 0), chunk_point_id(doc_b, 1, 0)
    await index.upsert(
        name, agent_a, [(pid_a, [0.1, 0.2, 0.3, 0.4], payload(agent_a, doc_a, pid_a, "a"))]
    )
    await index.upsert(
        name, agent_b, [(pid_b, [0.4, 0.3, 0.2, 0.1], payload(agent_b, doc_b, pid_b, "b"))]
    )

    assert await index.count(name, agent_a) == 1
    rows = await index.scroll_vectors(name, agent_a, limit=100)
    assert [r[0] for r in rows] == [str(pid_a)]
    assert len(rows[0][1]) == DIM
    assert await index.get_point(name, agent_a, pid_a) is not None
    assert await index.get_point(name, agent_a, pid_b) is None  # чужая точка = нет точки
    await qdrant.delete_collection(name)


async def test_upsert_rejects_payload_of_other_agent(
    qdrant: AsyncQdrantClient, collection_prefix: str
) -> None:
    index = QdrantChunkIndex(qdrant)
    name = collection_name(collection_prefix, "m", DIM)
    await index.ensure_collection(name, DIM)
    pid = uuid4()
    with pytest.raises(ValueError, match="agent_id"):
        await index.upsert(name, uuid4(), [(pid, [0.1] * DIM, payload(uuid4(), uuid4(), pid, "x"))])
    await qdrant.delete_collection(name)


async def test_upsert_is_idempotent_by_deterministic_ids(
    qdrant: AsyncQdrantClient, collection_prefix: str
) -> None:
    index = QdrantChunkIndex(qdrant)
    name = collection_name(collection_prefix, "m", DIM)
    await index.ensure_collection(name, DIM)
    agent, doc = uuid4(), uuid4()
    pid = chunk_point_id(doc, 1, 0)
    for _ in range(2):  # повторная доставка задачи
        point = (pid, [0.1, 0.2, 0.3, 0.4], payload(agent, doc, pid, "t"))
        await index.upsert(name, agent, [point])
    assert (await qdrant.count(name)).count == 1
    await qdrant.delete_collection(name)


async def _agent(db: Database, email: str) -> AgentOut:
    async with db.uow() as uow:
        owner = await UserRepository(uow.session).get_or_create(email)
        agent = await AgentRepository(uow.session).create_with_index(
            owner,
            name="A",
            slug=f"a-{uuid4().hex[:6]}",
            description="",
            persona_prompt=None,
            settings=AgentSettings(),
            embedding_model="m",
            dim=DIM,
            collection="c",
        )
        await uow.commit()
    return agent


async def test_repositories_hide_other_owners_data(db: Database) -> None:
    alice = await _agent(db, f"alice-{uuid4().hex[:6]}@t")
    bob = await _agent(db, f"bob-{uuid4().hex[:6]}@t")

    async with db.uow() as uow:
        doc, created = await DocumentRepository(uow.session).insert_if_new(
            document_id=uuid4(),
            agent_id=alice.id,
            filename="a.txt",
            fmt="txt",
            size_bytes=1,
            sha256=hashlib.sha256(uuid4().bytes).digest(),
            storage_key="k",
        )
        chats = ChatRepository(uow.session)
        chat = await chats.create_chat(alice.id, alice.owner_id, "q")
        msg = await chats.add_message(chat.id, MessageRole.ASSISTANT, MessageStatus.PENDING)
        await uow.commit()
    assert created

    async with db.session() as s:
        agents = AgentRepository(s)
        assert await agents.get(alice.owner_id, alice.id) is not None
        assert await agents.get(bob.owner_id, alice.id) is None
        assert [i.agent.id for i in await agents.list_for_owner(bob.owner_id)] == [bob.id]

        docs = DocumentRepository(s)
        assert await docs.get(alice.id, doc.id) is not None
        assert await docs.get(bob.id, doc.id) is None
        assert await docs.list_for_agent(bob.id) == []

        chats = ChatRepository(s)
        assert await chats.get_message(alice.id, alice.owner_id, msg.id) is not None
        assert await chats.get_message(bob.id, bob.owner_id, msg.id) is None
        assert await chats.get_message(alice.id, bob.owner_id, msg.id) is None
        assert await chats.get_chat(bob.id, bob.owner_id, chat.id) is None


async def test_same_file_is_deduplicated_per_agent_only(db: Database) -> None:
    a1 = await _agent(db, f"u1-{uuid4().hex[:6]}@t")
    a2 = await _agent(db, f"u2-{uuid4().hex[:6]}@t")
    sha = hashlib.sha256(b"same book").digest()
    results = []
    async with db.uow() as uow:
        repo = DocumentRepository(uow.session)
        for agent_id in (a1.id, a1.id, a2.id):
            _, created = await repo.insert_if_new(
                document_id=uuid4(),
                agent_id=agent_id,
                filename="b.txt",
                fmt="txt",
                size_bytes=9,
                sha256=sha,
                storage_key="k",
            )
            results.append(created)
        await uow.commit()
    assert results == [True, False, True]


async def test_claim_is_single_winner(db: Database) -> None:
    agent = await _agent(db, f"c-{uuid4().hex[:6]}@t")
    async with db.uow() as uow:
        doc, _ = await DocumentRepository(uow.session).insert_if_new(
            document_id=uuid4(),
            agent_id=agent.id,
            filename="c.txt",
            fmt="txt",
            size_bytes=1,
            sha256=hashlib.sha256(uuid4().bytes).digest(),
            storage_key="k",
        )
        await uow.commit()
    wins = []
    for _ in range(2):  # повторная доставка той же задачи
        async with db.uow() as uow:
            wins.append(await DocumentRepository(uow.session).claim(doc.id))
            await uow.commit()
    assert wins == [True, False]
