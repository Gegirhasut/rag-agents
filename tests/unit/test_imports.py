"""Все модули точек входа импортируются: ловит ошибки, видимые только в рантайме контейнера."""

import importlib

import pytest

ENTRYPOINT_MODULES = [
    "rag_agents.web.app",
    "rag_agents.workers.celery_app",
    "rag_agents.workers.runtime",
    "rag_agents.workers.tasks.ingest",
    "rag_agents.cli",
]


@pytest.mark.parametrize("module", ENTRYPOINT_MODULES)
def test_entrypoint_module_imports(module: str) -> None:
    importlib.import_module(module)


def test_ingest_task_is_registered_with_routing() -> None:
    from rag_agents.workers.celery_app import INGEST_TASK, celery_app  # noqa: PLC0415

    importlib.import_module("rag_agents.workers.tasks.ingest")
    assert INGEST_TASK in celery_app.tasks
    assert celery_app.conf.task_routes[INGEST_TASK]["queue"] == "ingest.parse"
