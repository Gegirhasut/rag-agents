"""Очередь maintenance: удаление документов и агентов, sweeper (beat)."""

from typing import Any

import structlog

from rag_agents.core.errors import TransientError
from rag_agents.domain.tasks import DeleteDocumentTask, PurgeAgentTask
from rag_agents.workers import runtime
from rag_agents.workers.celery_app import (
    DELETE_DOCUMENT_TASK,
    PURGE_AGENT_TASK,
    SWEEP_TASK,
    celery_app,
)
from rag_agents.workers.tasks.ingest import MAX_RETRIES, RETRY_BACKOFF, RETRY_BACKOFF_MAX

log = structlog.get_logger()


@celery_app.task(
    name=DELETE_DOCUMENT_TASK,
    autoretry_for=(TransientError,),
    retry_backoff=RETRY_BACKOFF,
    retry_backoff_max=RETRY_BACKOFF_MAX,
    retry_jitter=True,
    max_retries=MAX_RETRIES,
)
def delete_document(payload: dict[str, Any]) -> None:
    task = DeleteDocumentTask.model_validate(payload)
    runtime.run(runtime.container().ingest.delete_document(task))


@celery_app.task(
    name=PURGE_AGENT_TASK,
    autoretry_for=(TransientError,),
    retry_backoff=RETRY_BACKOFF,
    retry_backoff_max=RETRY_BACKOFF_MAX,
    retry_jitter=True,
    max_retries=MAX_RETRIES,
)
def purge_agent(payload: dict[str, Any]) -> None:
    task = PurgeAgentTask.model_validate(payload)
    runtime.run(runtime.container().ingest.purge_agent(task))


@celery_app.task(name=SWEEP_TASK, autoretry_for=(TransientError,), max_retries=0)
def sweep() -> None:
    runtime.run(runtime.container().ingest.sweep())
