"""Celery app и топология очередей (ARCHITECTURE §10).

Итерация 1: один воркер слушает все очереди. Аргументы очередей (DLX) задаются сразу:
у существующей очереди RabbitMQ их уже не поменять без пересоздания.
"""

from typing import Any

from celery import Celery
from kombu import Exchange, Queue

from rag_agents.core.config import get_settings
from rag_agents.domain.tasks import IngestDocumentTask

TASKS_EXCHANGE = Exchange("tasks", type="direct", durable=True)
DLX = Exchange("dlx", type="direct", durable=True)
QUEUE_NAMES = ("ingest.parse", "ingest.embed", "maintenance", "eval")
INGEST_TASK = "rag_agents.ingest_document"


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
    task_routes={INGEST_TASK: {"queue": "ingest.parse", "routing_key": "ingest.parse"}},
    task_acks_late=True,  # ack после выполнения: падение воркера → повторная доставка
    task_reject_on_worker_lost=True,
    task_acks_on_failure_or_timeout=False,  # необработанная ошибка → reject → DLX
    worker_prefetch_multiplier=1,
    task_time_limit=1500,
    task_soft_time_limit=1200,  # меньше consumer_timeout RabbitMQ (30 мин)
    worker_max_tasks_per_child=50,
    task_serializer="json",
    accept_content=["json"],
    task_ignore_result=True,
    broker_connection_retry_on_startup=True,
    worker_hijack_root_logger=False,
    # События задач (received/started/succeeded) — их читает Flower (профиль debug)
    worker_send_task_events=True,
    task_send_sent_event=True,
    include=["rag_agents.workers.tasks.ingest"],
)


class CeleryPublisher:
    """TaskPublisher для сервисов: публикует по имени задачи, не импортируя код воркера."""

    def publish_ingest(self, task: IngestDocumentTask) -> None:
        celery_app.send_task(INGEST_TASK, kwargs={"payload": task.model_dump(mode="json")})


def inspect_active(timeout: float = 0.5) -> dict[str, list[dict[str, Any]]]:
    """Что сейчас выполняют воркеры (broadcast через брокер; блокирует на timeout).

    Any: формат ответа Celery inspect — произвольный dict задачи.
    """
    result = celery_app.control.inspect(timeout=timeout).active() or {}
    return {worker: [dict(t) for t in tasks] for worker, tasks in result.items()}
