"""Auth, CSRF, API-ключи, problem+json, polling и rate limit — через HTTP всего приложения."""

import json
import re
from collections.abc import Callable
from uuid import UUID, uuid4

import pytest
from redis.asyncio import Redis

from rag_agents.domain.enums import DocumentStatus
from rag_agents.repositories.documents import DocumentRepository
from rag_agents.services.ratelimit import RateLimiter, Rule
from tests.integration.conftest import Session, Stack

pytestmark = pytest.mark.integration
Browser = Callable[[], Session]


async def test_pages_redirect_anonymous_to_login(stack: Stack) -> None:
    r = await stack.client.get("/agents/new")
    assert r.status_code == 303
    assert r.headers["location"] == "/login?next=/agents/new"

    # HTMX-запрос: вместо 303 — HX-Redirect на страницу, откуда шёл запрос
    r = await stack.client.get(
        f"/agents/{uuid4()}/documents/status",
        headers={"HX-Request": "true", "HX-Current-URL": "http://test/agents/x"},
    )
    assert r.status_code == 401
    assert r.headers["HX-Redirect"] == "/login?next=/agents/x"


async def test_login_sets_http_only_cookie_and_logout_ends_session(
    stack: Stack, browser: Browser
) -> None:
    user = await stack.user()
    b = browser()
    bad = await b.login(user.email, "wrong-password")
    assert bad.status_code == 401
    assert "Неверный email или пароль" in bad.text

    r = await b.login(user.email.upper())  # email без учёта регистра (citext)
    assert r.status_code == 303
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=lax" in cookie
    assert (await b.http.get("/")).status_code == 200

    # Выход без CSRF-токена не проходит
    assert (await b.http.post("/logout")).status_code == 403
    r = await b.http.post("/logout", data={"csrf_token": b.csrf})
    assert r.status_code == 303
    assert (await b.http.get("/")).status_code == 303


async def test_unsafe_requests_need_csrf_token(stack: Stack, browser: Browser) -> None:
    user = await stack.user()
    b = browser()
    await b.login(user.email)
    r = await b.http.post("/agents", data={"name": "без токена"})
    assert r.status_code == 403
    r = await b.http.post("/agents", data={"name": "с токеном", "csrf_token": b.csrf})
    assert r.status_code == 303
    r = await b.http.post("/agents", data={"name": "с заголовком"}, headers=b.hx())
    assert r.status_code == 303


async def test_api_requires_valid_key_and_speaks_problem_json(
    stack: Stack, browser: Browser
) -> None:
    r = await stack.client.get("/api/v1/agents")
    assert r.status_code == 401
    assert r.headers["content-type"] == "application/problem+json"
    assert r.headers["WWW-Authenticate"] == "Bearer"
    assert r.json()["code"] == "unauthorized"

    fake = {"Authorization": "Bearer rag_00000000_" + "a" * 43}
    assert (await stack.client.get("/api/v1/agents", headers=fake)).status_code == 401

    user = await stack.user()
    b = browser()
    await b.login(user.email)
    r = await b.http.post("/settings/api-keys", data={"name": "ci"}, headers=b.hx())
    assert r.status_code == 200
    token = _token_from(r.text)
    auth = {"Authorization": f"Bearer {token}"}

    r = await stack.client.post("/api/v1/agents", json={"name": "Толстой"}, headers=auth)
    assert r.status_code == 201
    agent = r.json()
    assert agent["slug"] == "tolstoi"
    # Bearer освобождён от CSRF, а чужой id — 404 в формате problem+json
    r = await stack.client.get(f"/api/v1/agents/{uuid4()}", headers=auth)
    assert (r.status_code, r.json()["code"]) == (404, "not_found")
    r = await stack.client.post("/api/v1/agents", json={"name": ""}, headers=auth)
    assert (r.status_code, r.json()["code"]) == (422, "validation_error")

    keys = await stack.client.get("/api/v1/me/api-keys", headers=auth)
    [key] = keys.json()
    assert "token" not in key
    assert key["last_used_at"] is not None
    r = await stack.client.delete(f"/api/v1/me/api-keys/{key['id']}", headers=auth)
    assert r.status_code == 200
    assert (await stack.client.get("/api/v1/agents", headers=auth)).status_code == 401


async def test_api_agent_documents_and_query_flow(stack: Stack) -> None:
    user = await stack.user()
    issued = await stack.container.auth.issue_key(user.id, "flow")
    auth = {"Authorization": f"Bearer {issued.token}"}
    api = stack.client

    agent = (await api.post("/api/v1/agents", json={"name": "Flow"}, headers=auth)).json()
    base = f"/api/v1/agents/{agent['id']}"
    r = await api.patch(base, json={"description": "новое"}, headers=auth)
    assert (r.status_code, r.json()["description"], r.json()["name"]) == (200, "новое", "Flow")

    n_tasks = len(stack.publisher.tasks)
    r = await api.post(
        f"{base}/documents",
        files=[("files", ("a.txt", "Текст книги".encode(), "text/plain"))],
        headers=auth,
    )
    assert r.status_code == 202
    [doc] = r.json()
    assert doc["status"] == "queued"
    assert len(stack.publisher.tasks) == n_tasks + 1  # задача опубликована после коммита

    r = await api.get(f"{base}/documents", params={"status": "done"}, headers=auth)
    assert r.json() == []
    r = await api.post(f"{base}/documents/{doc['id']}/retry", headers=auth)
    assert (r.status_code, r.json()["code"]) == (409, "conflict")
    r = await api.delete(f"{base}/documents/{doc['id']}", headers=auth)
    assert r.status_code == 202  # удаление асинхронное, можно и посреди обработки
    r = await api.get(f"{base}/documents", headers=auth)
    assert r.json() == []  # deleting в списке не показывается

    # Коллекция пустая → отказ «не нашёл» без LLM, но формат ответа полный
    r = await api.post(f"{base}/query", json={"question": "В чём смысл жизни?"}, headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["result"]["refused"] is True
    assert r.headers["X-Message-Id"] == body["message_id"]

    r = await api.post(
        f"{base}/query",
        json={"question": "Ещё раз?", "chat_id": body["chat_id"], "stream": True},
        headers=auth,
    )
    assert r.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert events[-1]["type"] == "done"
    assert r.headers["X-Chat-Id"] == body["chat_id"]

    assert (await api.delete(base, headers=auth)).status_code == 202
    assert (await api.get(base, headers=auth)).status_code == 404


async def test_status_polling_stops_with_286_when_all_terminal(
    stack: Stack, browser: Browser
) -> None:
    user = await stack.user()
    b = browser()
    await b.login(user.email)
    r = await b.http.post("/agents", data={"name": "Poll", "csrf_token": b.csrf})
    agent_path = r.headers["location"]
    agent_id = agent_path.rsplit("/", 1)[1]
    files = {"files": ("p.txt", "абзац".encode(), "text/plain")}
    await b.http.post(f"{agent_path}/documents", files=files, headers=b.hx())

    r = await b.http.get(f"{agent_path}/documents/status", headers=b.hx())
    assert r.status_code == 200
    assert 'hx-trigger="every 2s"' in r.text

    [doc] = await stack.container.documents.list(user.id, UUID(agent_id))
    async with stack.container.db.uow() as uow:
        await DocumentRepository(uow.session).mark_failed(doc.id, "broken", "Файл повреждён")
        await uow.commit()
    r = await b.http.get(f"{agent_path}/documents/status", headers=b.hx())
    assert r.status_code == 286
    assert "every 2s" not in r.text

    # Повтор из failed снова включает polling
    r = await b.http.post(f"{agent_path}/documents/{doc.id}/retry", headers=b.hx())
    assert r.status_code == 200
    assert 'hx-trigger="every 2s"' in r.text
    [doc] = await stack.container.documents.list(user.id, UUID(agent_id))
    assert doc.status == DocumentStatus.QUEUED


async def test_system_page_is_admin_only(stack: Stack, browser: Browser) -> None:
    user, admin = await stack.user(), await stack.user(admin=True)
    b, a = browser(), browser()
    await b.login(user.email)
    await a.login(admin.email)
    assert (await b.http.get("/system")).status_code == 404
    assert "Под капотом" not in (await b.http.get("/")).text
    assert (await a.http.get("/system")).status_code == 200


async def test_sliding_window_rate_limit(redis: Redis) -> None:
    limiter = RateLimiter(redis)
    rule = Rule(f"t{uuid4().hex[:6]}", limit=3, window_s=60)
    decisions = [await limiter.hit(rule, "u1") for _ in range(4)]
    assert [d.allowed for d in decisions] == [True, True, True, False]
    assert [d.remaining for d in decisions[:3]] == [2, 1, 0]
    assert 0 < decisions[3].retry_after_s <= 60
    assert (await limiter.hit(rule, "u2")).allowed  # счётчик на субъекта


async def test_questions_over_limit_get_429(stack: Stack, monkeypatch: pytest.MonkeyPatch) -> None:
    user = await stack.user()
    issued = await stack.container.auth.issue_key(user.id, "rl")
    auth = {"Authorization": f"Bearer {issued.token}"}
    agent = (await stack.client.post("/api/v1/agents", json={"name": "RL"}, headers=auth)).json()
    monkeypatch.setattr(stack.container.settings, "rl_questions_per_min", 1)
    url = f"/api/v1/agents/{agent['id']}/query"
    assert (await stack.client.post(url, json={"question": "раз?"}, headers=auth)).is_success
    r = await stack.client.post(url, json={"question": "два?"}, headers=auth)
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) > 0
    assert r.json()["code"] == "rate_limited"


async def test_openapi_documents_only_json_api(stack: Stack) -> None:
    spec = (await stack.client.get("/api/v1/openapi.json")).json()
    paths = set(spec["paths"])
    assert "/api/v1/agents/{agent_id}/query" in paths
    assert "/agents/new" not in paths
    assert "HTTPBearer" in spec["components"]["securitySchemes"]


def _token_from(html: str) -> str:
    m = re.search(r"(rag_[0-9a-f]{8}_[A-Za-z0-9_-]+)", html)
    assert m, "токен не показан"
    return m.group(1)
