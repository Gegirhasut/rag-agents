import os
import re
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import httpx
import pytest
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from rag_agents.container import Container, build_container
from rag_agents.core.config import Settings
from rag_agents.core.db import Database
from rag_agents.core.observability import NoopTracer
from rag_agents.domain.auth import UserOut
from rag_agents.llm.prices import PriceTable
from rag_agents.rag.chunking.structural import StructuralChunker
from rag_agents.services.ingest import IngestService
from rag_agents.services.query import QueryService
from rag_agents.web.app import create_app
from tests.fakes import VECTOR, FakeEmbedder, FakeLLM, RecordingPublisher, WordCounter

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+asyncpg://rag_test:rag_test@127.0.0.1:15432/rag_test"
)
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:16333")


@pytest.fixture(scope="session")
async def db() -> AsyncIterator[Database]:
    database = Database(DATABASE_URL, pool_size=2)
    yield database
    await database.dispose()


@pytest.fixture(scope="session")
async def qdrant() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(url=QDRANT_URL)
    yield client
    await client.close()


@pytest.fixture
def collection_prefix() -> str:
    return f"t{uuid.uuid4().hex[:8]}_"


REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:16379/0")


@pytest.fixture(scope="session")
async def redis() -> AsyncIterator[Redis]:
    client: Redis = Redis.from_url(REDIS_URL, decode_responses=True)
    await client.flushdb()
    yield client
    await client.aclose()


# --- HTTP-уровень: всё приложение через ASGI, хранилища настоящие, LLM и очередь — фейки ---


@dataclass
class Stack:
    client: httpx.AsyncClient
    transport: httpx.ASGITransport
    container: Container
    publisher: RecordingPublisher

    async def drain(self) -> int:
        """Исполняет опубликованные задачи, как воркеры, пока очереди не опустеют."""
        p, ingest, n = self.publisher, self.container.ingest, 0
        while p.tasks or p.embeds or p.deletes or p.purges:
            if p.tasks:
                await ingest.parse(p.tasks.pop(0), final_attempt=False)
            elif p.embeds:
                await ingest.embed_batch(p.embeds.pop(0), final_attempt=False)
            elif p.deletes:
                await ingest.delete_document(p.deletes.pop(0))
            else:
                await ingest.purge_agent(p.purges.pop(0))
            n += 1
        return n

    async def user(self, *, admin: bool = False) -> UserOut:
        email = f"u-{uuid.uuid4().hex[:8]}@test.local"
        return await self.container.auth.create_user(email, PASSWORD, is_admin=admin)


PASSWORD = "test-password-1"  # только для эфемерной тестовой БД


@pytest.fixture(scope="session")
async def stack(tmp_path_factory: pytest.TempPathFactory) -> AsyncIterator[Stack]:
    settings = Settings(
        app_env="test",
        database_url=DATABASE_URL,
        redis_url=REDIS_URL,
        qdrant_url=QDRANT_URL,
        qdrant_collection_prefix=f"t{uuid.uuid4().hex[:8]}_",
        embedding_model="fake-embed",
        embedding_dim=len(VECTOR),
        upload_dir=tmp_path_factory.mktemp("uploads"),
        trace_enabled=False,
        # Логин из тестов идёт с одного «IP»; лимиты проверяются отдельными тестами
        rl_login_per_min=10_000,
        rl_questions_per_min=10_000,
        rl_uploads_per_hour=10_000,
        embedding_batch_size=4,  # несколько батчей даже на маленьких фикстурах
    )
    publisher = RecordingPublisher()

    def factory(_: Settings) -> Container:
        c = build_container(settings, publisher)
        # Вопросы без Ollama и DeepSeek: тот же QueryService, но с фейковыми эмбеддером и LLM
        c.query = QueryService(
            c.db,
            FakeEmbedder(),
            c.query.index,
            FakeLLM(),
            c.trace,
            settings,
            NoopTracer(),
            PriceTable({}),
        )
        # Ingest как в воркере, но с детерминированным счётчиком токенов и фейковым эмбеддером
        c._ingest = IngestService(
            c.db,
            c.documents.storage,
            lambda: StructuralChunker(
                WordCounter(), target=40, max_tokens=60, min_tokens=5, overlap=8
            ),
            FakeEmbedder(),
            c.query.index,
            c.documents.progress,
            c.trace,
            settings,
            NoopTracer(),
            publisher,
        )
        return c

    app = create_app(factory)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield Stack(client, transport, app.state.container, publisher)


class Session:
    """Браузер одного пользователя: свои cookie и CSRF-токен."""

    def __init__(self, stack: Stack) -> None:
        self.http = httpx.AsyncClient(transport=stack.transport, base_url="http://test")
        self.csrf = ""

    async def login(self, email: str, password: str = PASSWORD) -> httpx.Response:
        r = await self.http.post("/login", data={"email": email, "password": password})
        if r.status_code == 303:
            page = await self.http.get("/settings/api-keys")
            m = re.search(r'name="csrf-token" content="([^"]+)"', page.text)
            assert m, "на странице нет csrf-token"
            self.csrf = m.group(1)
        return r

    def hx(self) -> dict[str, str]:
        return {"HX-Request": "true", "X-CSRF-Token": self.csrf}

    async def aclose(self) -> None:
        await self.http.aclose()


@pytest.fixture
async def browser(stack: Stack) -> AsyncIterator[Callable[[], Session]]:
    opened: list[Session] = []

    def make() -> Session:
        s = Session(stack)
        opened.append(s)
        return s

    yield make
    for s in opened:
        await s.aclose()
