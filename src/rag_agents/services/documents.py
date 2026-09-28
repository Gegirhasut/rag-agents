import builtins
from collections.abc import AsyncIterator
from pathlib import PurePath
from typing import Protocol
from uuid import UUID

import anyio
import structlog

from rag_agents.core.config import Settings
from rag_agents.core.db import Database
from rag_agents.core.ids import uuid7
from rag_agents.core.storage import FileTooLargeError, LocalFileStorage
from rag_agents.domain.documents import DocumentOut
from rag_agents.domain.enums import DocumentStatus, SourceFormat
from rag_agents.domain.tasks import IngestDocumentTask
from rag_agents.rag.index.qdrant import QdrantChunkIndex
from rag_agents.repositories.agents import AgentRepository
from rag_agents.repositories.documents import DocumentRepository
from rag_agents.services.errors import ConflictError, NotFoundError, ValidationError
from rag_agents.services.progress import ProgressStore
from rag_agents.services.trace import TraceBus

log = structlog.get_logger()

# Итерация 1 — только txt; остальные форматы появятся в итерации 3
SUPPORTED: dict[str, SourceFormat] = {".txt": SourceFormat.TXT}
PLANNED = {".fb2", ".epub", ".pdf", ".docx"}


class TaskPublisher(Protocol):
    def publish_ingest(self, task: IngestDocumentTask) -> None: ...


def detect_format(filename: str) -> SourceFormat:
    suffix = PurePath(filename).suffix.lower()
    if suffix in SUPPORTED:
        return SUPPORTED[suffix]
    if suffix in PLANNED:
        raise ValidationError(
            f"Формат {suffix} появится в следующих итерациях; сейчас — только .txt"
        )
    raise ValidationError(f"Неподдерживаемый формат: {suffix or 'без расширения'}")


class DocumentService:
    def __init__(
        self,
        db: Database,
        storage: LocalFileStorage,
        progress: ProgressStore,
        publisher: TaskPublisher,
        trace: TraceBus,
        settings: Settings,
        index: QdrantChunkIndex,
    ) -> None:
        self.db = db
        self.index = index
        self.storage = storage
        self.progress = progress
        self.publisher = publisher
        self.trace = trace
        self.settings = settings

    async def upload(
        self, owner_id: UUID, agent_id: UUID, filename: str, chunks: AsyncIterator[bytes]
    ) -> tuple[DocumentOut, bool]:
        """Потоковая загрузка → документ queued → задача в очередь ПОСЛЕ коммита."""
        fmt = detect_format(filename)
        async with self.db.session() as s:
            agent = await AgentRepository(s).get(owner_id, agent_id)
        if agent is None or agent.active_index_id is None:
            raise NotFoundError("agent")
        index_id = agent.active_index_id
        await self.trace.emit(
            "upload.received",
            "browser",
            "web",
            f"HTMX: POST multipart «{PurePath(filename).name[:60]}»",
            agent_id=agent_id,
        )

        document_id = uuid7()
        try:
            stored = await self.storage.save_stream(
                agent_id, document_id, fmt.value, chunks, self.settings.max_upload_bytes
            )
        except FileTooLargeError as e:
            raise ValidationError(f"Файл больше {self.settings.max_upload_mb} МБ") from e
        if stored.size == 0:
            await self.storage.delete(stored.key)
            raise ValidationError("Файл пустой")
        await self.trace.emit(
            "upload.stored",
            "web",
            "uploads",
            f"Файл потоком записан на диск ({stored.size // 1024} КБ, sha256)",
            agent_id=agent_id,
        )

        async with self.db.uow() as uow:
            doc, created = await DocumentRepository(uow.session).insert_if_new(
                document_id=document_id,
                agent_id=agent_id,
                filename=PurePath(filename).name[:255],
                fmt=fmt.value,
                size_bytes=stored.size,
                sha256=stored.sha256,
                storage_key=stored.key,
            )
            if created:
                task = IngestDocumentTask(document_id=doc.id, index_id=index_id)
                uow.on_commit(
                    lambda: self.trace.emit(
                        "pg.document",
                        "web",
                        "postgres",
                        "COMMIT: INSERT documents (status=queued)",
                        agent_id=agent_id,
                    )
                )
                uow.on_commit(lambda: anyio.to_thread.run_sync(self.publisher.publish_ingest, task))
                uow.on_commit(
                    lambda: self.trace.emit(
                        "celery.published",
                        "web",
                        "rabbitmq",
                        "После COMMIT: задача ingest_document (только ID) → очередь ingest.parse",
                        agent_id=agent_id,
                    )
                )
            await uow.commit()
        if not created:
            await self.storage.delete(stored.key)
            await self.trace.emit(
                "pg.duplicate",
                "web",
                "postgres",
                "UNIQUE (agent_id, sha256): такой файл уже есть — копия удалена, задачи нет",
                agent_id=agent_id,
            )
            log.info("document.duplicate", agent_id=str(agent_id), document_id=str(doc.id))
        else:
            log.info("document.queued", agent_id=str(agent_id), document_id=str(doc.id))
        return doc, created

    async def list(
        self, owner_id: UUID, agent_id: UUID, status: DocumentStatus | None = None
    ) -> list[DocumentOut]:
        """Статусы — из PG одним запросом, прогресс в работе — из Redis (ARCHITECTURE §6.4)."""
        async with self.db.session() as s:
            if await AgentRepository(s).get(owner_id, agent_id) is None:
                raise NotFoundError("agent")
            docs = await DocumentRepository(s).list_for_agent(agent_id)
        if status is not None:
            docs = [d for d in docs if d.status == status]
        return await self._with_progress(docs)

    async def get(self, owner_id: UUID, agent_id: UUID, document_id: UUID) -> DocumentOut:
        async with self.db.session() as s:
            if await AgentRepository(s).get(owner_id, agent_id) is None:
                raise NotFoundError("agent")
            doc = await DocumentRepository(s).get(agent_id, document_id)
        if doc is None:
            raise NotFoundError("document")
        [doc] = await self._with_progress([doc])
        return doc

    async def _with_progress(self, docs: builtins.list[DocumentOut]) -> builtins.list[DocumentOut]:
        hot = await self.progress.get_many([d.id for d in docs if not d.status.is_terminal])
        return [
            d.model_copy(update={"progress": 100 if d.status.is_terminal else hot.get(d.id, 0)})
            for d in docs
        ]

    async def retry(self, owner_id: UUID, agent_id: UUID, document_id: UUID) -> DocumentOut:
        """Повтор обработки только из failed: задача публикуется после коммита."""
        async with self.db.uow() as uow:
            agent = await AgentRepository(uow.session).get(owner_id, agent_id)
            if agent is None or agent.active_index_id is None:
                raise NotFoundError("agent")
            repo = DocumentRepository(uow.session)
            doc = await repo.reset_failed(agent_id, document_id)
            if doc is None:
                if await repo.get(agent_id, document_id) is None:
                    raise NotFoundError("document")
                raise ConflictError("Повторить можно только документ в статусе failed")
            task = IngestDocumentTask(document_id=doc.id, index_id=agent.active_index_id)
            uow.on_commit(lambda: anyio.to_thread.run_sync(self.publisher.publish_ingest, task))
            await uow.commit()
        log.info("document.retry", agent_id=str(agent_id), document_id=str(document_id))
        return doc

    async def delete(self, owner_id: UUID, agent_id: UUID, document_id: UUID) -> None:
        """Синхронное удаление документа в конечном статусе: точки Qdrant → строка PG
        (чанки каскадом) → файл. corpus_version растёт: кэши ответов станут невалидны.

        Документ в работе удалять нельзя (409): воркер допишет точки после удаления.
        Асинхронное удаление через статус deleting — итерация 3.
        """
        async with self.db.session() as s:
            agents = AgentRepository(s)
            agent = await agents.get(owner_id, agent_id)
            if agent is None:
                raise NotFoundError("agent")
            doc = await DocumentRepository(s).get(agent_id, document_id)
            if doc is None:
                raise NotFoundError("document")
            if not doc.status.is_terminal:
                raise ConflictError("Документ ещё обрабатывается — удалить можно после завершения")
            idx = (
                await agents.get_index(agent_id, agent.active_index_id)
                if agent.active_index_id
                else None
            )
        if idx is not None:
            # Сначала Qdrant: если он недоступен, документ остаётся целым, можно повторить
            await self.index.delete_document(idx.collection, agent_id, document_id)
        async with self.db.uow() as uow:
            if not await DocumentRepository(uow.session).delete_terminal(agent_id, document_id):
                raise ConflictError("Статус документа изменился, повторите")
            await AgentRepository(uow.session).bump_corpus_version(agent_id)
            await uow.commit()
        await self.storage.delete(doc.storage_key)
        log.info("document.deleted", agent_id=str(agent_id), document_id=str(document_id))
