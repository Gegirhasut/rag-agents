"""Composition root: единственное место, где собираются зависимости.

Web и воркеры получают готовые сервисы и не импортируют rag/llm/repositories напрямую.
"""

from dataclasses import dataclass, field
from urllib.parse import unquote, urlsplit

import httpx
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from rag_agents.core.config import Settings
from rag_agents.core.db import Database
from rag_agents.core.storage import LocalFileStorage
from rag_agents.llm.openai_compat import OpenAICompatProvider
from rag_agents.rag.chunking.naive import NaiveChunker
from rag_agents.rag.chunking.tokenizer import load_token_counter
from rag_agents.rag.embeddings.ollama import OllamaEmbedder
from rag_agents.rag.index.qdrant import QdrantChunkIndex
from rag_agents.services.agents import AgentService
from rag_agents.services.documents import DocumentService, TaskPublisher
from rag_agents.services.ingest import IngestService
from rag_agents.services.progress import ProgressStore
from rag_agents.services.query import QueryService
from rag_agents.services.system import SystemService
from rag_agents.services.trace import TraceBus
from rag_agents.workers.celery_app import CeleryPublisher, inspect_active


@dataclass
class Container:
    settings: Settings
    db: Database
    redis: Redis
    qdrant: AsyncQdrantClient
    ollama_http: httpx.AsyncClient
    rabbit_http: httpx.AsyncClient
    llm: OpenAICompatProvider
    agents: AgentService
    documents: DocumentService
    query: QueryService
    trace: TraceBus
    system: SystemService
    _ingest: IngestService | None = field(default=None)

    @property
    def ingest(self) -> IngestService:
        if self._ingest is None:
            raise RuntimeError("ingest service is built only in workers (with_ingest=True)")
        return self._ingest

    async def aclose(self) -> None:
        await self.llm.aclose()
        await self.system.aclose()
        await self.rabbit_http.aclose()
        await self.ollama_http.aclose()
        await self.qdrant.close()
        await self.redis.aclose()
        await self.db.dispose()


def preload_worker_resources(settings: Settings) -> None:
    """Тяжёлые read-only ресурсы грузятся в родителе Celery до fork (общие страницы, CoW)."""
    load_token_counter(settings.tokenizer_path)


def build_container(
    settings: Settings,
    publisher: TaskPublisher | None = None,
    *,
    with_ingest: bool = False,
    db_pool_size: int = 5,
) -> Container:
    if publisher is None:
        publisher = CeleryPublisher()
    db = Database(settings.database_url, pool_size=db_pool_size)
    redis: Redis = Redis.from_url(settings.redis_url, decode_responses=True)
    qdrant = AsyncQdrantClient(url=settings.qdrant_url, timeout=30)
    # Эмбеддинг батча на CPU может занимать десятки секунд
    ollama_http = httpx.AsyncClient(
        base_url=settings.ollama_base_url, timeout=httpx.Timeout(300, connect=5)
    )
    index = QdrantChunkIndex(qdrant)
    embedder = OllamaEmbedder(
        ollama_http, settings.embedding_model, settings.embedding_dim, settings.embedding_num_thread
    )
    storage = LocalFileStorage(settings.upload_dir)
    progress = ProgressStore(redis)
    api_key = settings.deepseek_api_key.get_secret_value() if settings.deepseek_api_key else None
    llm = OpenAICompatProvider(
        name="deepseek",
        base_url=settings.llm_base_url,
        api_key=api_key,
        model=settings.llm_model,
        reasoning_effort=settings.llm_reasoning_effort,
        connect_timeout_s=settings.llm_connect_timeout_s,
        read_timeout_s=settings.llm_read_timeout_s,
    )
    trace = TraceBus(redis, enabled=settings.trace_enabled)
    broker = urlsplit(settings.celery_broker_url)
    rabbit_http = httpx.AsyncClient(
        base_url=settings.rabbitmq_management_url,
        auth=(unquote(broker.username or "guest"), unquote(broker.password or "guest")),
        timeout=2,
    )
    ingest = None
    if with_ingest:
        chunker = NaiveChunker(
            load_token_counter(settings.tokenizer_path),
            target=settings.chunk_target_tokens,
            max_tokens=settings.chunk_max_tokens,
        )
        ingest = IngestService(db, storage, chunker, embedder, index, progress, trace, settings)
    return Container(
        settings=settings,
        db=db,
        redis=redis,
        qdrant=qdrant,
        ollama_http=ollama_http,
        rabbit_http=rabbit_http,
        llm=llm,
        agents=AgentService(db, index, settings),
        documents=DocumentService(db, storage, progress, publisher, trace, settings),
        query=QueryService(db, embedder, index, llm, trace, settings),
        trace=trace,
        system=SystemService(db, redis, index, ollama_http, rabbit_http, inspect_active, settings),
        _ingest=ingest,
    )
