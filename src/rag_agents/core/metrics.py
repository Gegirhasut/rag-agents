"""Метрики Prometheus (ARCHITECTURE §14.3).

Web (2 процесса uvicorn) и воркеры Celery (prefork) — разные процессы, поэтому
prometheus_client работает в multiprocess mode: каждый процесс пишет свои значения в mmap-файлы
каталога PROMETHEUS_MULTIPROC_DIR (общий volume), а `/metrics` в web суммирует их все.
Без переменной окружения (unit-тесты, локальный запуск) метрики живут в памяти процесса.

Метки — только с ограниченным набором значений (шаблон роута, а не путь с id; провайдер,
а не текст ошибки), иначе число временных рядов растёт без предела.
"""

import os
import socket
from collections.abc import Iterable

import structlog
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
    multiprocess,
    values,
)
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

log = structlog.get_logger()


def process_id() -> str:
    """Имя процесса в файлах multiprocess-метрик: hostname контейнера + PID.

    По умолчанию prometheus_client называет файлы по PID, а у каждого контейнера своё
    пространство PID: PID 1 есть и в web, и в воркерах, и в beat, и все они писали бы в один
    counter_1.db общего volume. Без «_» — по нему коллектор разбирает имя файла.
    """
    return f"{socket.gethostname().replace('_', '-')}-{os.getpid()}"


# До создания первой метрики: класс значения выбирается при конструировании метрики
if values.ValueClass is not values.MutexValue:
    # prometheus_client.values не аннотирован
    values.ValueClass = values.MultiProcessValue(process_identifier=process_id)  # type: ignore[no-untyped-call]

_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 21, 34, 60)

HTTP_REQUESTS = Counter("http_requests_total", "HTTP-запросы", ["route", "method", "status"])
HTTP_SECONDS = Histogram(
    "http_request_seconds",
    "Длительность HTTP-запроса (для SSE — до конца стрима)",
    ["route", "method"],
    buckets=_LATENCY_BUCKETS,
)
RAG_STAGE_SECONDS = Histogram(
    "rag_stage_seconds",
    "Этапы ответа: embed_query, search, context, ttft, total",
    ["stage"],
    buckets=_LATENCY_BUCKETS,
)
RAG_ANSWERS = Counter(
    "rag_answers_total", "Ответы", ["refused", "grounded", "cache_hit", "provider", "fallback"]
)
LLM_TOKENS = Counter("llm_tokens_total", "Токены LLM", ["provider", "model", "kind"])
LLM_COST = Counter("llm_cost_usd_total", "Стоимость LLM, $", ["provider", "model"])
LLM_ERRORS = Counter("llm_errors_total", "Ошибки LLM", ["provider", "code"])
INGEST_DOCUMENTS = Counter(
    "ingest_documents_total", "Документы, завершившие обработку", ["status", "format"]
)
INGEST_SECONDS = Histogram(
    "ingest_duration_seconds",
    "Индексация документа от claim до done",
    ["format"],
    buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600),
)
INGEST_CHUNKS = Counter("ingest_chunks_total", "Проиндексированные чанки")
EMBED_BATCH_SECONDS = Histogram(
    "embed_batch_seconds", "Эмбеддинг одного батча (Ollama)", buckets=_LATENCY_BUCKETS
)


class QueueDepthCollector(Collector):
    """celery_queue_depth{queue}: глубины очередей RabbitMQ на момент scrape.

    Значения снимает web перед рендером (management API, async); коллектор только отдаёт их.
    """

    def __init__(self, depths: dict[str, int]) -> None:
        self._depths = depths

    def collect(self) -> Iterable[GaugeMetricFamily]:
        family = GaugeMetricFamily(
            "celery_queue_depth", "Сообщений в очереди RabbitMQ", labels=["queue"]
        )
        for queue, depth in sorted(self._depths.items()):
            family.add_metric([queue], depth)
        yield family


def render(extra: Iterable[Collector] = ()) -> tuple[bytes, str]:
    """Текст экспозиции для /metrics: все процессы (multiprocess) + дополнительные коллекторы."""
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)  # type: ignore[no-untyped-call]
        body = generate_latest(registry)
    else:
        body = generate_latest(REGISTRY)
    own = CollectorRegistry()
    for collector in extra:
        own.register(collector)
    return body + generate_latest(own), CONTENT_TYPE_LATEST
