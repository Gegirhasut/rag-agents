"""Контракты страницы «Под капотом» (/system): живые события пайплайна и снимок состояния сервисов.

Dev-инструмент для наглядности (ARCHITECTURE §14.4). Тексты вопросов и документов в события
не попадают — только размеры, тайминги, скоры и метаданные книг.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

# Узлы схемы на странице /system: id совпадают с data-node в шаблоне
Node = Literal[
    "browser",
    "web",
    "postgres",
    "redis",
    "rabbitmq",
    "uploads",
    "worker",
    "qdrant",
    "ollama",
    "llm",
]


class TraceEvent(BaseModel):
    """Одно «движение» по схеме: src → dst с подписью. dst=None — событие внутри узла."""

    ts: float  # unix time, секунды
    kind: str  # query.search, ingest.embed_batch, …
    src: Node
    dst: Node | None = None
    label: str
    agent_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class ServiceStatus(BaseModel):
    ok: bool = True
    error: str | None = None


class QdrantCollectionStats(BaseModel):
    name: str
    status: str
    points: int
    indexed_vectors: int
    segments: int
    vectors: list[str]  # "dense: 1024 × Cosine, on_disk" …
    sparse_vectors: list[str]
    hnsw: str
    quantization: str | None
    payload_indexes: list[str]
    per_agent: list[tuple[str, int]]  # (имя агента, точек)


class QdrantStats(ServiceStatus):
    version: str | None = None
    collections: list[QdrantCollectionStats] = Field(default_factory=list)


class QueueStats(BaseModel):
    name: str
    ready: int
    unacked: int
    consumers: int
    publish_rate: float
    deliver_rate: float
    is_dlq: bool


class WorkerTask(BaseModel):
    name: str
    id: str
    runtime_s: float | None
    args: str


class CeleryStats(ServiceStatus):
    queues: list[QueueStats] = Field(default_factory=list)
    workers: dict[str, list[WorkerTask]] | None = None  # None — inspect не ответил
    rabbit_version: str | None = None
    connections: int = 0


class RedisKey(BaseModel):
    key: str
    kind: str
    value: str


class RedisStats(ServiceStatus):
    version: str | None = None
    used_memory: str | None = None
    max_memory: str | None = None
    clients: int = 0
    ops_per_sec: int = 0
    keyspace: dict[str, str] = Field(default_factory=dict)
    sample_keys: list[RedisKey] = Field(default_factory=list)


class TableStats(BaseModel):
    name: str
    rows: int
    size: str


class PostgresStats(ServiceStatus):
    version: str | None = None
    db_size: str | None = None
    tables: list[TableStats] = Field(default_factory=list)


class OllamaModel(BaseModel):
    name: str
    family: str | None
    parameters: str | None
    quantization: str | None
    size: str
    loaded: bool


class OllamaStats(ServiceStatus):
    base_url: str
    models: list[OllamaModel] = Field(default_factory=list)


class LlmInfo(BaseModel):
    base_url: str
    model: str
    reasoning_effort: str | None
    key_configured: bool


class AdminUi(BaseModel):
    key: str
    title: str
    port: int
    path: str
    about: str
    up: bool


class SystemSnapshot(BaseModel):
    qdrant: QdrantStats
    celery: CeleryStats
    redis: RedisStats
    postgres: PostgresStats
    ollama: OllamaStats
    llm: LlmInfo
    admin_uis: list[AdminUi]


class VectorPoint(BaseModel):
    id: str
    x: float
    y: float
    doc: int  # индекс в VectorMap.docs
    ord: int
    chapter: str | None


class VectorMapDoc(BaseModel):
    id: str
    title: str
    points: int


class VectorMap(BaseModel):
    """2D-проекция (PCA) всех векторов агента + базис, чтобы браузер спроецировал вопрос."""

    agent_id: str
    collection: str
    dim: int
    total: int  # точек агента в коллекции
    points: list[VectorPoint]
    docs: list[VectorMapDoc]
    mean: list[float]
    components: list[list[float]]  # 2 × dim
    explained: list[float]  # доля дисперсии по осям


class PointDetail(BaseModel):
    id: str
    book_title: str | None
    chapter_title: str | None
    ord: int
    text: str
    vector: list[float]
    norm: float
