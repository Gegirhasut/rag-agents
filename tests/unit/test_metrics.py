"""Prometheus: экспозиция, глубина очередей, метка роута без id."""

import os
import re
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from rag_agents.core import metrics
from rag_agents.web.middleware import route_label


def test_render_includes_app_metrics_and_queue_depth() -> None:
    metrics.RAG_ANSWERS.labels("false", "true", "false", "deepseek", "false").inc()
    body, content_type = metrics.render(
        [metrics.QueueDepthCollector({"ingest.embed": 3, "ingest.embed.dlq": 0})]
    )
    text = body.decode()
    assert content_type.startswith("text/plain")
    assert 'rag_answers_total{cache_hit="false",fallback="false",grounded="true"' in text
    assert 'celery_queue_depth{queue="ingest.embed"} 3.0' in text
    assert "# TYPE rag_stage_seconds histogram" in text


def test_route_label_uses_template_not_path() -> None:
    route = SimpleNamespace(
        path="/agents/{agent_id}", path_regex=re.compile(r"^/agents/(?P<agent_id>[^/]+)$")
    )
    assert route_label({"route": route, "path": "/agents/0190"}) == "/agents/{agent_id}"
    # Роут вложенного роутера: префикс /api/v1 восстанавливается из пути
    assert (
        route_label({"route": route, "path": "/api/v1/agents/0190"}) == "/api/v1/agents/{agent_id}"
    )
    assert route_label({"path": "/static/app.css"}) == "/static"
    assert route_label({"path": "/wp-login.php"}) == "unmatched"


def test_multiprocess_files_are_unique_per_container(tmp_path: Path) -> None:
    """Регрессия 2026-09-29: web, воркеры и beat делят volume метрик, а PID у каждого
    контейнера свой (PID 1 есть везде) → несколько процессов писали в один counter_1.db."""
    code = "from rag_agents.core import metrics\nmetrics.INGEST_CHUNKS.inc()\n"
    env = {**os.environ, "PROMETHEUS_MULTIPROC_DIR": str(tmp_path)}
    subprocess.run([sys.executable, "-c", code], env=env, check=True)  # noqa: S603 — свой код
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names
    assert all(f"_{socket.gethostname()}-" in n for n in names), names
