"""/metrics: HTTP-метрики с шаблоном роута и метрики ответа после вопроса через API."""

import re

import pytest

from tests.integration.conftest import Stack

pytestmark = pytest.mark.integration


def _value(text: str, name: str, **labels: str) -> float:
    """Значение ряда с заданными метками (порядок меток в экспозиции — алфавитный)."""
    for line in text.splitlines():
        if not line.startswith(name + "{"):
            continue
        # rindex: в значении метки route тоже есть «}» ({agent_id})
        got = dict(re.findall(r'(\w+)="([^"]*)"', line[: line.rindex("}")]))
        if all(got.get(k) == v for k, v in labels.items()):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


async def test_metrics_count_requests_by_route_template_and_answers(stack: Stack) -> None:
    before = (await stack.client.get("/metrics")).text
    user = await stack.user()
    issued = await stack.container.auth.issue_key(user.id, "metrics")
    auth = {"Authorization": f"Bearer {issued.token}"}
    agent = (await stack.client.post("/api/v1/agents", json={"name": "M"}, headers=auth)).json()
    r = await stack.client.post(
        f"/api/v1/agents/{agent['id']}/query", json={"question": "Что там?"}, headers=auth
    )
    assert r.status_code == 200

    resp = await stack.client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    text = resp.text
    route = "/api/v1/agents/{agent_id}/query"
    labels = {"route": route, "method": "POST", "status": "200"}
    assert (
        _value(text, "http_requests_total", **labels)
        == _value(before, "http_requests_total", **labels) + 1
    )
    assert agent["id"] not in text  # id не попадает в метки
    # Пустой корпус → отказ без LLM
    refused = {"refused": "true", "provider": "none"}
    assert _value(text, "rag_answers_total", **refused) >= 1
    assert _value(text, "rag_stage_seconds_count", stage="embed_query") >= 1
    assert "celery_queue_depth" in text  # без RabbitMQ — пустое семейство, а не 500
