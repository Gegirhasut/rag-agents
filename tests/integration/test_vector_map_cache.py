"""Карту векторов строит воркер, web только отдаёт готовую из Redis (общую для процессов).

Сборка — это ~40 МБ JSON с векторами из Qdrant и PCA: в web она на десятки секунд занимала
CPU и GIL процесса, и он переставал отвечать на другие запросы.
"""

from uuid import UUID, uuid4

import httpx
import pytest
from qdrant_client import AsyncQdrantClient, QdrantClient
from redis.asyncio import Redis

from rag_agents.core.config import Settings
from rag_agents.core.db import Database
from rag_agents.domain.documents import ChunkPayload
from rag_agents.domain.tasks import BuildVectorMapTask
from rag_agents.rag.index.qdrant import QdrantChunkIndex
from rag_agents.services.system import SystemService
from tests.fakes import RecordingPublisher
from tests.integration.conftest import QDRANT_URL
from tests.integration.test_query_tracing import _agent_with_chunk

pytestmark = pytest.mark.integration


class _CountingIndex(QdrantChunkIndex):
    def __init__(self, client: AsyncQdrantClient) -> None:
        super().__init__(client, bulk_client=QdrantClient(url=QDRANT_URL))
        self.scrolls = 0

    async def scroll_vectors(
        self, collection: str, agent_id: UUID, limit: int
    ) -> list[tuple[str, list[float], ChunkPayload]]:
        self.scrolls += 1
        return await super().scroll_vectors(collection, agent_id, limit)


def _no_workers() -> dict[str, list[dict[str, object]]]:
    return {}


def _service(
    db: Database, redis: Redis, index: QdrantChunkIndex, publisher: RecordingPublisher
) -> SystemService:
    http = httpx.AsyncClient()
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    return SystemService(db, redis, index, http, http, _no_workers, settings, publisher)


async def test_vector_map_is_built_by_worker_and_shared_between_web_processes(
    db: Database, qdrant: AsyncQdrantClient, redis: Redis, collection_prefix: str
) -> None:
    agent = await _agent_with_chunk(db, qdrant, collection_prefix, f"vm-{uuid4().hex[:6]}@t")
    publisher = RecordingPublisher()
    # два процесса uvicorn и воркер = три экземпляра сервиса с общим Redis
    web1, web2, worker = (_CountingIndex(qdrant) for _ in range(3))

    assert await _service(db, redis, web1, publisher).vector_map(agent.owner_id, agent.id) is None
    assert await _service(db, redis, web2, publisher).vector_map(agent.owner_id, agent.id) is None
    # пока карта строится, повторные запросы не плодят задачи
    assert publisher.vector_maps == [BuildVectorMapTask(agent_id=agent.id)]

    await _service(db, redis, worker, publisher).build_vector_map(publisher.vector_maps[0])
    built = await _service(db, redis, web1, publisher).vector_map(agent.owner_id, agent.id)
    again = await _service(db, redis, web2, publisher).vector_map(agent.owner_id, agent.id)

    assert built is not None
    assert built == again
    assert built.total == 1
    assert (web1.scrolls, web2.scrolls, worker.scrolls) == (0, 0, 1)
    assert len(publisher.vector_maps) == 1
