"""Задачи ingest: тонкие sync-обёртки, вся логика — в IngestService (ADR-7)."""

from typing import Any

from celery import Task

from rag_agents.core.errors import TransientError
from rag_agents.domain.tasks import EmbedBatchTask, ParseTask
from rag_agents.workers import runtime
from rag_agents.workers.celery_app import EMBED_TASK, PARSE_TASK, celery_app

MAX_RETRIES = 6
# Экспоненциальный backoff 5 → 300 с с jitter: ~10 мин суммарно (§10.3)
RETRY_BACKOFF = 5
RETRY_BACKOFF_MAX = 300


@celery_app.task(
    name=PARSE_TASK,
    bind=True,
    autoretry_for=(TransientError,),
    retry_backoff=RETRY_BACKOFF,
    retry_backoff_max=RETRY_BACKOFF_MAX,
    retry_jitter=True,
    max_retries=MAX_RETRIES,
)
def parse_document(self: "Task[Any, Any]", payload: dict[str, Any]) -> None:
    task = ParseTask.model_validate(payload)
    runtime.run(
        runtime.container().ingest.parse(
            task, final_attempt=self.request.retries >= MAX_RETRIES, attempt=self.request.retries
        )
    )


@celery_app.task(
    name=EMBED_TASK,
    bind=True,
    autoretry_for=(TransientError,),
    retry_backoff=RETRY_BACKOFF,
    retry_backoff_max=RETRY_BACKOFF_MAX,
    retry_jitter=True,
    max_retries=MAX_RETRIES,
)
def embed_batch(self: "Task[Any, Any]", payload: dict[str, Any]) -> None:
    task = EmbedBatchTask.model_validate(payload)
    runtime.run(
        runtime.container().ingest.embed_batch(
            task, final_attempt=self.request.retries >= MAX_RETRIES, attempt=self.request.retries
        )
    )
