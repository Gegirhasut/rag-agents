"""Event loop на процесс воркера (ADR-7).

Celery prefork выполняет одну задачу на процесс за раз. Корутины сервисов исполняются
в постоянном loop процесса: asyncio.run() на задачу пересоздавал бы loop и ломал пулы
соединений (asyncpg, httpx), привязанные к loop.
"""

import asyncio
import os
from collections.abc import Coroutine
from typing import Any

import structlog
from celery.signals import (
    task_postrun,
    task_prerun,
    worker_init,
    worker_process_init,
    worker_process_shutdown,
)
from kombu import Connection

from rag_agents.container import Container, build_container, preload_worker_resources
from rag_agents.core.config import get_settings
from rag_agents.core.logging import configure_logging
from rag_agents.workers.celery_app import REQUEST_ID_HEADER, CeleryPublisher, dlq_queues

_loop: asyncio.AbstractEventLoop | None = None
_container: Container | None = None


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    if _loop is None:
        raise RuntimeError("worker runtime is not initialized")
    return _loop.run_until_complete(coro)


def container() -> Container:
    if _container is None:
        raise RuntimeError("worker runtime is not initialized")
    return _container


@worker_init.connect
def _init_parent(**_: Any) -> None:
    """Родитель prefork, до fork детей.

    DLX и *.dlq Celery сам не объявляет (он не должен их слушать) — объявляем при старте.
    Тяжёлые read-only ресурсы грузим здесь, чтобы дети делили их страницы (copy-on-write).
    """
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    with Connection(settings.celery_broker_url) as conn:
        channel = conn.default_channel
        for q in dlq_queues():
            q.bind(channel).declare()
    # Токенизатор нужен только парсингу: worker-embed (WORKER_ROLE=embed) его не грузит
    if settings.worker_role != "embed":
        preload_worker_resources(settings)


@worker_process_init.connect
def _init_process(**_: Any) -> None:
    global _loop, _container  # noqa: PLW0603  состояние процесса воркера
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    try:
        _container = build_container(settings, CeleryPublisher(), with_ingest=True, db_pool_size=2)
    except Exception:
        # Celery логирует исключения сигналов и продолжает работу: процесс объявил бы себя
        # готовым, но падал бы на каждой задаче. Лучше громко умереть — это видно в логах.
        structlog.get_logger().exception("worker.init_failed")
        os._exit(1)


@worker_process_shutdown.connect
def _shutdown_process(**_: Any) -> None:
    if _loop is not None and _container is not None:
        _loop.run_until_complete(_container.aclose())
        _loop.close()


@task_prerun.connect
def _bind_task_context(task_id: str | None = None, task: Any = None, **_: Any) -> None:
    """task_id и request_id (заголовок сообщения из web) — в каждую строку лога задачи.

    Any: сигнатура сигналов Celery.
    """
    structlog.contextvars.clear_contextvars()
    request_id = getattr(getattr(task, "request", None), REQUEST_ID_HEADER, None)
    structlog.contextvars.bind_contextvars(
        task_id=task_id,
        task=getattr(task, "name", None),
        **({"request_id": request_id} if request_id else {}),
    )


@task_postrun.connect
def _clear_task_context(**_: Any) -> None:
    structlog.contextvars.clear_contextvars()
