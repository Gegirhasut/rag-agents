from uuid import UUID

from pydantic import BaseModel


class IngestDocumentTask(BaseModel):
    """Сообщение очереди — только идентификаторы (ARCHITECTURE §5)."""

    document_id: UUID
    index_id: UUID
