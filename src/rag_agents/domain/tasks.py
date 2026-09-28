"""Сообщения очередей — только идентификаторы, никакого контента (ARCHITECTURE §5, §10)."""

from uuid import UUID

from pydantic import BaseModel


class ParseTask(BaseModel):
    """ingest.parse: парсинг → чанкинг → запись чанков → фан-аут батчей эмбеддинга."""

    document_id: UUID
    job_id: UUID
    index_id: UUID


class EmbedBatchTask(BaseModel):
    """ingest.embed: эмбеддинг и upsert одного батча чанков."""

    document_id: UUID
    job_id: UUID
    index_id: UUID
    batch_no: int


class DeleteDocumentTask(BaseModel):
    """maintenance: удалить точки Qdrant, строки PG и файл документа."""

    document_id: UUID
    agent_id: UUID


class PurgeAgentTask(BaseModel):
    """maintenance: очистить данные мягко удалённого агента."""

    agent_id: UUID
