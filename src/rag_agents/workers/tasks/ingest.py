from typing import Any

import structlog
from celery import Task

from rag_agents.core.errors import TransientError
from rag_agents.domain.tasks import IngestDocumentTask
from rag_agents.workers import runtime
from rag_agents.workers.celery_app import INGEST_TASK, celery_app

log = structlog.get_logger()
MAX_RETRIES = 6


@celery_app.task(
    name=INGEST_TASK,
    bind=True,
    autoretry_for=(TransientError,),
    retry_backoff=5,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=MAX_RETRIES,
)
def ingest_document(self: "Task[Any, Any]", payload: dict[str, Any]) -> None:
    """Тонкая sync-обёртка: вся логика — в IngestService."""
    task = IngestDocumentTask.model_validate(payload)
    structlog.contextvars.bind_contextvars(task_id=self.request.id)
    try:
        runtime.run(
            runtime.container().ingest.ingest(
                task,
                final_attempt=self.request.retries >= MAX_RETRIES,
                attempt=self.request.retries,
            )
        )
    finally:
        structlog.contextvars.unbind_contextvars("task_id")
