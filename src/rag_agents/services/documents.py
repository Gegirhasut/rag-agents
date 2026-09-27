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
from rag_agents.domain.enums import SourceFormat
from rag_agents.domain.tasks import IngestDocumentTask
from rag_agents.repositories.agents import AgentRepository
from rag_agents.repositories.documents import DocumentRepository
from rag_agents.services.errors import NotFoundError, ValidationError
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
    ) -> None:
        self.db = db
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

    async def list(self, owner_id: UUID, agent_id: UUID) -> list[DocumentOut]:
        async with self.db.session() as s:
            if await AgentRepository(s).get(owner_id, agent_id) is None:
                raise NotFoundError("agent")
            docs = await DocumentRepository(s).list_for_agent(agent_id)
        hot = await self.progress.get_many([d.id for d in docs if not d.status.is_terminal])
        return [
            d.model_copy(update={"progress": 100 if d.status.is_terminal else hot.get(d.id, 0)})
            for d in docs
        ]
