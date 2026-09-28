"""Изоляция данных агентов (SPEC NFR «Изоляция»): ни один путь чтения не видит чужое."""

import hashlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
from qdrant_client import AsyncQdrantClient

from rag_agents.core.db import Database
from rag_agents.domain.agents import AgentCreate, AgentOut, AgentSettings
from rag_agents.domain.documents import ChunkPayload
from rag_agents.domain.enums import MessageRole, MessageStatus
from rag_agents.rag.index.qdrant import QdrantChunkIndex, chunk_point_id, collection_name
from rag_agents.repositories.agents import AgentRepository
from rag_agents.repositories.chats import ChatRepository
from rag_agents.repositories.documents import DocumentRepository
from rag_agents.repositories.insights import InsightsRepository
from rag_agents.repositories.users import UserRepository
from tests.integration.conftest import Session, Stack

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


async def test_insights_facts_are_scoped_to_owner(db: Database) -> None:
    """Сводка «Аналитики» строится по PG: чужие ответы и документы в неё не попадают."""
    alice = await _agent(db, f"ins-a-{uuid4().hex[:6]}@t")
    bob = await _agent(db, f"ins-b-{uuid4().hex[:6]}@t")
    since = datetime.now(UTC) - timedelta(minutes=5)
    async with db.uow() as uow:
        chats = ChatRepository(uow.session)
        chat = await chats.create_chat(alice.id, alice.owner_id, "q")
        msg = await chats.add_message(chat.id, MessageRole.ASSISTANT, MessageStatus.PENDING)
        await chats.finish_message(
            msg.id,
            status=MessageStatus.DONE,
            content="ответ",
            usage={"provider": "deepseek", "model": "deepseek-flash", "reasoning_effort": None,
                   "t_retrieval_ms": 100, "t_total_ms": 900, "cost_usd": 0.001},
        )  # fmt: skip
        docs = DocumentRepository(uow.session)
        doc, _ = await docs.insert_if_new(
            document_id=uuid4(),
            agent_id=alice.id,
            filename="a.txt",
            fmt="txt",
            size_bytes=1,
            sha256=hashlib.sha256(uuid4().bytes).digest(),
            storage_key="k",
        )
        await docs.mark_done(doc.id, {"parse_ms": 12, "embed_ms": 3400})
        await uow.commit()

    async with db.session() as s:
        repo = InsightsRepository(s)
        [a] = await repo.answers(alice.owner_id, since)
        assert (a.agent_id, a.usage.cost_usd if a.usage else None) == (alice.id, 0.001)
        [d] = await repo.documents(alice.owner_id, since)
        assert d.timings == {"parse_ms": 12, "embed_ms": 3400}
        assert await repo.answers(bob.owner_id, since) == []
        assert await repo.documents(bob.owner_id, since) == []


# --- HTTP: обход всех эндпоинтов с чужими ID (CLAUDE.md: новые эндпоинты добавляются сюда) ---

# (метод, путь, канал). Плейсхолдеры подставляются ID данных Алисы; запрос делает Боб
# (админ — чтобы /system не отсекался раньше проверки владельца).
FOREIGN_CASES: list[tuple[str, str, str]] = [
    ("GET", "/agents/{agent}", "web"),
    ("POST", "/agents/{agent}/documents", "web"),
    ("GET", "/agents/{agent}/documents/status", "web"),
    ("POST", "/agents/{agent}/documents/{doc}/retry", "web"),
    ("DELETE", "/agents/{agent}/documents/{doc}", "web"),
    ("POST", "/agents/{agent}/chats/new/messages", "web"),
    ("POST", "/agents/{agent}/chats/{chat}/messages", "web"),
    ("GET", "/agents/{agent}/messages/{msg}/stream", "web"),
    ("POST", "/agents/{agent}/messages/{msg}/feedback", "web"),
    ("GET", "/system/agents/{agent}/vector-map", "web"),
    ("GET", "/system/agents/{agent}/points/{point}", "web"),
    ("DELETE", "/settings/api-keys/{key}", "web"),
    ("GET", "/api/v1/agents/{agent}", "api"),
    ("PATCH", "/api/v1/agents/{agent}", "api"),
    ("DELETE", "/api/v1/agents/{agent}", "api"),
    ("POST", "/api/v1/agents/{agent}/documents", "api"),
    ("GET", "/api/v1/agents/{agent}/documents", "api"),
    ("GET", "/api/v1/agents/{agent}/documents/{doc}", "api"),
    ("POST", "/api/v1/agents/{agent}/documents/{doc}/retry", "api"),
    ("DELETE", "/api/v1/agents/{agent}/documents/{doc}", "api"),
    ("POST", "/api/v1/agents/{agent}/query", "api"),
    ("DELETE", "/api/v1/me/api-keys/{key}", "api"),
]
# Эндпоинты с параметром пути, которые здесь не обходятся, и почему
EXEMPT = {
    # Трейсы живут в Langfuse (в тестах выключен); чужой трейс → 404 проверяет
    # tests/unit/test_insights.py::test_foreign_trace_is_not_found_even_from_cache
    ("GET", "/insights/traces/{trace_id}"),
    ("GET", "/insights/sessions/{session_id}"),
}
_PLACEHOLDER_NAMES = {
    "agent_id": "agent",
    "document_id": "doc",
    "chat_ref": "chat",
    "message_id": "msg",
    "point_id": "point",
    "key_id": "key",
}


def _declared_routes() -> set[tuple[str, str]]:
    """Все роуты с параметрами пути из листовых роутеров (web + API)."""
    from fastapi.routing import APIRoute  # noqa: PLC0415

    from rag_agents.api.v1 import agents as api_agents  # noqa: PLC0415
    from rag_agents.api.v1 import documents as api_documents  # noqa: PLC0415
    from rag_agents.api.v1 import keys as api_keys  # noqa: PLC0415
    from rag_agents.api.v1 import query as api_query  # noqa: PLC0415
    from rag_agents.web.routes import (  # noqa: PLC0415
        auth,
        chat,
        documents,
        insights,
        pages,
        system,
    )

    web = [auth, chat, documents, insights, pages, system]
    api = [api_agents, api_documents, api_keys, api_query]
    out: set[tuple[str, str]] = set()
    for prefix, modules in (("", web), ("/api/v1", api)):
        for m in modules:
            for r in m.router.routes:
                if isinstance(r, APIRoute) and "{" in r.path:
                    out |= {(method, prefix + r.path) for method in r.methods}
    return out


def _normalize(path: str) -> str:
    for param, short in _PLACEHOLDER_NAMES.items():
        path = path.replace("{" + short + "}", "{" + param + "}")
    return path.replace("/chats/new/", "/chats/{chat_ref}/")


def test_every_parametrized_endpoint_is_walked() -> None:
    covered = {(m, _normalize(p)) for m, p, _ in FOREIGN_CASES} | EXEMPT
    missing = _declared_routes() - covered
    assert not missing, f"добавьте в FOREIGN_CASES: {sorted(missing)}"


@pytest.fixture(scope="module")
async def alice_data(stack: Stack) -> dict[str, str]:
    c = stack.container
    alice = await stack.user(admin=True)  # чтобы контроль владельца прошёл и через /system
    agent = await c.agents.create(alice.id, AgentCreate(name="Алиса"))
    doc, _ = await c.documents.upload(alice.id, agent.id, "a.txt", _bytes("секрет Алисы"))
    pair = await c.query.ask(alice.id, agent.id, None, "Что там?")
    key = await c.auth.issue_key(alice.id, "alice")
    return {
        "user": str(alice.id),
        "email": alice.email,
        "token": key.token,
        "agent": str(agent.id),
        "doc": str(doc.id),
        "chat": str(pair.answer.chat_id),
        "msg": str(pair.answer.id),
        "point": str(uuid4()),
        "key": str(key.id),
    }


@pytest.fixture(scope="module")
async def bob(stack: Stack) -> AsyncIterator[tuple[Session, dict[str, str]]]:
    user = await stack.user(admin=True)
    session = Session(stack)
    await session.login(user.email)
    key = await stack.container.auth.issue_key(user.id, "bob")
    yield session, {"Authorization": f"Bearer {key.token}"}
    await session.aclose()


def _bytes(text: str) -> AsyncIterator[bytes]:
    async def gen() -> AsyncIterator[bytes]:
        yield text.encode()

    return gen()


def _request_kwargs(method: str, path: str) -> dict[str, Any]:
    if path.endswith("/documents") and method == "POST":
        return {"files": {"files": ("b.txt", b"bob", "text/plain")}}
    if path.endswith("/messages"):
        return {"data": {"question": "Покажи чужое"}}
    if path.endswith("/feedback"):
        return {"data": {"value": "1"}}
    if path.endswith("/query"):
        return {"json": {"question": "Покажи чужое"}}
    if method == "PATCH":
        return {"json": {"name": "взлом"}}
    return {}


@pytest.mark.parametrize(("method", "path", "channel"), FOREIGN_CASES)
async def test_foreign_ids_are_not_found(
    stack: Stack,
    alice_data: dict[str, str],
    bob: tuple[Session, dict[str, str]],
    method: str,
    path: str,
    channel: str,
) -> None:
    session, bearer = bob
    url = path.format(**alice_data)
    kwargs = _request_kwargs(method, path)
    if channel == "web":
        r = await session.http.request(method, url, headers=session.hx(), **kwargs)
    else:
        r = await stack.client.request(method, url, headers=bearer, **kwargs)
    assert r.status_code == 404, f"{method} {url} → {r.status_code}: {r.text[:200]}"
    if channel == "api":
        assert r.headers["content-type"] == "application/problem+json"


async def test_owner_reaches_same_endpoints(stack: Stack, alice_data: dict[str, str]) -> None:
    """Контроль: 404 выше — из-за владельца, а не из-за опечатки в пути."""
    session = Session(stack)
    await session.login(alice_data["email"])
    bearer = {"Authorization": f"Bearer {alice_data['token']}"}
    try:
        for method, path, channel in FOREIGN_CASES:
            if method != "GET" or "{point}" in path or "/stream" in path:
                continue
            url = path.format(**alice_data)
            if channel == "web":
                r = await session.http.get(url, headers=session.hx())
            else:
                r = await stack.client.get(url, headers=bearer)
            assert r.status_code == 200, f"{url} → {r.status_code}"
    finally:
        await session.aclose()


async def test_alice_data_survives_bobs_attempts(stack: Stack, alice_data: dict[str, str]) -> None:
    """После обхода (порядок тестов модуля) данные Алисы на месте и без изменений."""
    c = stack.container
    alice, agent_id = UUID(alice_data["user"]), UUID(alice_data["agent"])
    agent = await c.agents.get(alice, agent_id)
    assert agent.name == "Алиса"
    docs = await c.documents.list(alice, agent_id)
    assert [str(d.id) for d in docs] == [alice_data["doc"]]
    [key] = await c.auth.list_keys(alice)
    assert key.revoked_at is None
