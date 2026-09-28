"""Celery app и топология очередей (ARCHITECTURE §10).

worker-ingest слушает ingest.parse, maintenance и eval; worker-embed — ingest.embed; beat
раз в минуту запускает sweeper. У рабочих очередей DLX: reject → <queue>.dlq.
"""

from typing import Any

import structlog
from celery import Celery
from kombu import Exchange, Queue

from rag_agents.core.config import get_settings
from rag_agents.domain.tasks import DeleteDocumentTask, EmbedBatchTask, ParseTask, PurgeAgentTask

TASKS_EXCHANGE = Exchange("tasks", type="direct", durable=True)
DLX = Exchange("dlx", type="direct", durable=True)
QUEUE_NAMES = ("ingest.parse", "ingest.embed", "maintenance", "eval")
PARSE_TASK = "rag_agents.ingest.parse"
EMBED_TASK = "rag_agents.ingest.embed"
DELETE_DOCUMENT_TASK = "rag_agents.maintenance.delete_document"
PURGE_AGENT_TASK = "rag_agents.maintenance.purge_agent"
SWEEP_TASK = "rag_agents.maintenance.sweep"
SWEEP_EVERY_S = 60.0
REQUEST_ID_HEADER = "request_id"


def _route(queue: str) -> dict[str, str]:
    return {"queue": queue, "routing_key": queue}


def _work_queue(name: str) -> Queue:
    return Queue(
        name,
        TASKS_EXCHANGE,
        routing_key=name,
        durable=True,
        queue_arguments={
            "x-dead-letter-exchange": DLX.name,
            "x-dead-letter-routing-key": f"{name}.dlq",
        },
    )


def dlq_queues() -> list[Queue]:
    return [Queue(f"{n}.dlq", DLX, routing_key=f"{n}.dlq", durable=True) for n in QUEUE_NAMES]


settings = get_settings()
celery_app = Celery("rag_agents", broker=settings.celery_broker_url)
celery_app.conf.update(
    task_queues=[_work_queue(n) for n in QUEUE_NAMES],
    task_default_queue="maintenance",
    task_default_exchange=TASKS_EXCHANGE.name,
    task_routes={
        PARSE_TASK: _route("ingest.parse"),
        EMBED_TASK: _route("ingest.embed"),
        DELETE_DOCUMENT_TASK: _route("maintenance"),
        PURGE_AGENT_TASK: _route("maintenance"),
        SWEEP_TASK: _route("maintenance"),
    },
    beat_schedule={
        # Без expires: Celery делает из него TTL сообщения, и RabbitMQ отправляет истёкший
        # sweep в maintenance.dlq, пока worker-ingest занят парсингом большой книги.
        # Лишний прогон sweeper безвреден — он идемпотентен.
        "sweep": {"task": SWEEP_TASK, "schedule": SWEEP_EVERY_S},
    },
    task_acks_late=True,  # ack после выполнения: падение воркера → повторная доставка
    task_reject_on_worker_lost=True,
    task_acks_on_failure_or_timeout=False,  # необработанная ошибка → reject → DLX
    worker_prefetch_multiplier=1,
    # Порог перезапуска ребёнка по памяти задаётся флагом --max-memory-per-child в compose.
    # Инициализация ребёнка (контейнер зависимостей, Langfuse) под нагрузкой CPU занимает
    # больше дефолтных 4 с: Celery убивал детей SIGKILL-ом в цикле, задачи не выполнялись
    worker_proc_alive_timeout=60,
    task_time_limit=1500,
    task_soft_time_limit=1200,  # меньше consumer_timeout RabbitMQ (30 мин)
    worker_max_tasks_per_child=50,
    task_serializer="json",
    accept_content=["json"],
    task_ignore_result=True,
    broker_connection_retry_on_startup=True,
    worker_hijack_root_logger=False,
    # structlog пишет JSON в stdout сам; иначе Celery заворачивает строки в WARNING redirected
    worker_redirect_stdouts=False,
    # События задач (received/started/succeeded) — их читает Flower (профиль debug)
    worker_send_task_events=True,
    task_send_sent_event=True,
    include=["rag_agents.workers.tasks.ingest", "rag_agents.workers.tasks.maintenance"],
)


def _headers() -> dict[str, str]:
    """request_id запроса, породившего задачу: сквозной ключ в логах web и воркеров."""
    rid = structlog.contextvars.get_contextvars().get("request_id")
    return {REQUEST_ID_HEADER: str(rid)} if rid else {}


class CeleryPublisher:
    """TaskPublisher для сервисов: публикует по имени задачи, не импортируя код воркера."""

    @staticmethod
    def _send(name: str, payload: dict[str, Any]) -> None:
        celery_app.send_task(name, kwargs={"payload": payload}, headers=_headers())

    def publish_parse(self, task: ParseTask) -> None:
        self._send(PARSE_TASK, task.model_dump(mode="json"))

    def publish_embed(self, tasks: list[EmbedBatchTask]) -> None:
        # Одно соединение с брокером на весь фан-аут (десятки сообщений)
        with celery_app.producer_or_acquire() as producer:
            for t in tasks:
                celery_app.send_task(
                    EMBED_TASK,
                    kwargs={"payload": t.model_dump(mode="json")},
                    headers=_headers(),
                    producer=producer,
                )

    def publish_delete_document(self, task: DeleteDocumentTask) -> None:
        self._send(DELETE_DOCUMENT_TASK, task.model_dump(mode="json"))

    def publish_purge_agent(self, task: PurgeAgentTask) -> None:
        self._send(PURGE_AGENT_TASK, task.model_dump(mode="json"))


def inspect_active(timeout: float = 0.5) -> dict[str, list[dict[str, Any]]]:
    """Что сейчас выполняют воркеры (broadcast через брокер; блокирует на timeout).

    Any: формат ответа Celery inspect — произвольный dict задачи.
    """
    result = celery_app.control.inspect(timeout=timeout).active() or {}
    return {worker: [dict(t) for t in tasks] for worker, tasks in result.items()}
