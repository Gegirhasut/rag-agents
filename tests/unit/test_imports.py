"""Все модули точек входа импортируются: ловит ошибки, видимые только в рантайме контейнера."""

import importlib

import pytest

ENTRYPOINT_MODULES = [
    "rag_agents.web.app",
    "rag_agents.workers.celery_app",
    "rag_agents.workers.runtime",
    "rag_agents.workers.tasks.ingest",
    "rag_agents.workers.tasks.maintenance",
    "rag_agents.workers.dlq",
    "rag_agents.cli",
]


@pytest.mark.parametrize("module", ENTRYPOINT_MODULES)
def test_entrypoint_module_imports(module: str) -> None:
    importlib.import_module(module)


@pytest.mark.parametrize(
    ("task", "queue"),
    [
        ("rag_agents.ingest.parse", "ingest.parse"),
        ("rag_agents.ingest.embed", "ingest.embed"),
        ("rag_agents.maintenance.delete_document", "maintenance"),
        ("rag_agents.maintenance.purge_agent", "maintenance"),
        ("rag_agents.maintenance.sweep", "maintenance"),
    ],
)
def test_tasks_are_registered_with_routing(task: str, queue: str) -> None:
    from rag_agents.workers.celery_app import celery_app  # noqa: PLC0415

    importlib.import_module("rag_agents.workers.tasks.ingest")
    importlib.import_module("rag_agents.workers.tasks.maintenance")
    assert task in celery_app.tasks
    assert celery_app.conf.task_routes[task]["queue"] == queue


def test_work_queues_dead_letter_to_dlx() -> None:
    from rag_agents.workers.celery_app import celery_app  # noqa: PLC0415

    for q in celery_app.conf.task_queues:
        assert q.queue_arguments == {
            "x-dead-letter-exchange": "dlx",
            "x-dead-letter-routing-key": f"{q.name}.dlq",
        }


def test_sweep_schedule_has_no_message_ttl() -> None:
    """Истёкшее сообщение RabbitMQ шлёт в DLX: периодические задачи без expires."""
    from rag_agents.workers.celery_app import celery_app  # noqa: PLC0415

    for entry in celery_app.conf.beat_schedule.values():
        assert "expires" not in entry.get("options", {})


def test_child_init_timeout_survives_loaded_cpu() -> None:
    """Дефолт Celery 4 с: под нагрузкой инициализация ребёнка дольше, и Celery убивал
    детей SIGKILL-ом по кругу, задачи не выполнялись (SPEC §9, «что пошло не так»)."""
    from rag_agents.workers.celery_app import celery_app  # noqa: PLC0415

    assert celery_app.conf.worker_proc_alive_timeout >= 30
