from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from rag_agents.domain.enums import DocumentStatus, IngestStage, SourceFormat


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    agent_id: UUID
    filename: str
    format: SourceFormat
    size_bytes: int
    storage_key: str
    status: DocumentStatus
    stage: IngestStage | None
    progress: int = 0  # из Redis (ARCHITECTURE §6.4), в PG не хранится
    error_code: str | None
    error_message: str | None
    title: str | None
    chunks_total: int | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @property
    def duration_s(self) -> float | None:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None


class ChunkDraft(BaseModel):
    """Результат чанкинга до записи в БД (ARCHITECTURE §5)."""

    ord: int
    section_path: list[str]
    chapter_title: str | None
    text: str
    embed_text: str
    token_count: int
    char_start: int
    char_end: int


class ChunkPayload(BaseModel):
    """Payload точки в Qdrant. agent_id — tenant-ключ."""

    agent_id: str
    document_id: str
    chunk_id: str
    ord: int
    book_title: str | None
    author: str | None
    section_path: list[str]
    chapter_title: str | None
    page_from: int | None = None
    page_to: int | None = None
    text: str
