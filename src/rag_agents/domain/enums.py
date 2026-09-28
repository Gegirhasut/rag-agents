from enum import StrEnum


class DocumentStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
    DELETING = "deleting"

    @property
    def is_terminal(self) -> bool:
        return self in (DocumentStatus.DONE, DocumentStatus.FAILED)


class IngestStage(StrEnum):
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    FINALIZING = "finalizing"


class SourceFormat(StrEnum):
    TXT = "txt"
    FB2 = "fb2"
    EPUB = "epub"
    PDF = "pdf"
    DOCX = "docx"


class IndexStatus(StrEnum):
    BUILDING = "building"
    ACTIVE = "active"
    RETIRED = "retired"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class MessageStatus(StrEnum):
    PENDING = "pending"
    STREAMING = "streaming"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
