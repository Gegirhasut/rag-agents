"""E2E smoke по живому стенду (критерии итерации 1, docs/PLAN.md).

Логин → агент → загрузка txt → ждёт done → вопрос → SSE-стрим (те же HTML/HTMX-эндпоинты,
что и браузер) → API-ключ → тот же вопрос через JSON API со стримом (критерии итерации 2).

    uv run python scripts/smoke.py [--base http://localhost:8080] [--file path.txt]
"""

import argparse
import html
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
from web_session import login

DEFAULT_FILE = Path("data/samples/Л. Н. Толстой - Исповедь.txt")
QUESTION = "В чём смысл жизни?"
# Критерий PLAN: ≤ 5 мин
INGEST_TIMEOUT_S = int(os.environ.get("SMOKE_INGEST_TIMEOUT_S", "300"))
EXPECTED_SOURCES = 6


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def strip_tags(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8080")
    ap.add_argument("--file", type=Path, default=DEFAULT_FILE)
    ap.add_argument("--question", default=QUESTION)
    args = ap.parse_args()
    if not args.file.exists():
        fail(f"нет файла {args.file} (см. README, «Как проверить итерацию 1 руками», шаг 4)")

    c = httpx.Client(base_url=args.base, timeout=httpx.Timeout(120, connect=5))

    ready = c.get("/readyz").json()
    print(f"readyz: {ready}")
    if not ready["ready"]:
        fail("стенд не готов")

    anon = c.get("/")
    if anon.status_code != 303 or not anon.headers["location"].startswith("/login"):
        fail(f"без входа / должен вести на /login, а не HTTP {anon.status_code}")
    login(c)
    print("вход: сессия и CSRF-токен получены")

    name = f"Smoke {time.strftime('%H:%M:%S')}"
    r = c.post("/agents", data={"name": name, "description": "e2e smoke"})
    if r.status_code != 303:
        fail(f"создание агента: HTTP {r.status_code}")
    agent_path = r.headers["location"]
    print(f"агент: {name} → {agent_path}")

    t_upload = time.monotonic()
    with args.file.open("rb") as f:
        r = c.post(
            f"{agent_path}/documents",
            files={"files": (args.file.name, f, "text/plain")},
            headers={"HX-Request": "true"},
        )
    if r.status_code != 200 or "alert-danger" in r.text:
        fail(f"загрузка: HTTP {r.status_code} {strip_tags(r.text)[:200]}")

    status, body, seen = "", "", []
    while time.monotonic() - t_upload < INGEST_TIMEOUT_S:
        r = c.get(f"{agent_path}/documents/status", headers={"HX-Request": "true"})
        body = r.text
        m = re.search(r'data-status="(\w+)"', body)
        status = m.group(1) if m else "?"
        progress = re.search(r"width: (\d+)%", body)
        seen.append((status, progress.group(1) if progress else ""))
        if r.status_code == 286:  # HTMX stop polling: все документы в конечном статусе
            break
        time.sleep(2)
    ingest_s = time.monotonic() - t_upload
    transitions = [s for i, s in enumerate(seen) if i == 0 or s != seen[i - 1]]
    print(f"статусы: {' → '.join(f'{s}{f" {p}%" if p else ""}' for s, p in transitions)}")
    if status != "done":
        fail(f"документ не done за {ingest_s:.0f} с: {status} {strip_tags(body)[-200:]}")
    chunks = re.search(r"(\d+) чанков", body)
    print(f"ingest: done за {ingest_s:.1f} с, чанков: {chunks.group(1) if chunks else '?'}")

    with args.file.open("rb") as f:
        r = c.post(
            f"{agent_path}/documents",
            files={"files": (args.file.name, f, "text/plain")},
            headers={"HX-Request": "true"},
        )
    if "уже загружен" not in r.text:
        fail("повторная загрузка того же файла не распознана как дубликат")
    print("дедупликация: повторная загрузка пропущена")

    r = c.post(
        f"{agent_path}/chats/new/messages",
        data={"question": args.question},
        headers={"HX-Request": "true"},
    )
    m = re.search(r'sse-connect="([^"]+)"', r.text)
    if r.status_code != 200 or not m:
        fail(f"вопрос: HTTP {r.status_code}")
    stream_url = m.group(1)

    t0 = time.monotonic()
    ttft = None
    events: dict[str, int] = {}
    sources_html = done_html = ""
    tokens: list[str] = []
    with c.stream("GET", stream_url) as resp:
        if resp.headers.get("content-type", "").split(";")[0] != "text/event-stream":
            fail(f"stream content-type: {resp.headers.get('content-type')}")
        event, data_lines = None, []
        for line in resp.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data_lines.append(line[6:])
            elif line == "" and event:
                data = "\n".join(data_lines)
                events[event] = events.get(event, 0) + 1
                if event == "token":
                    ttft = ttft or time.monotonic() - t0
                    tokens.append(html.unescape(data))
                elif event == "sources":
                    sources_html = data
                elif event == "done":
                    done_html = data
                    break
                event, data_lines = None, []
    total = time.monotonic() - t0

    n_sources = sources_html.count('class="source-card"')
    print(f"SSE-события: {events}")
    print(f"источников: {n_sources}; первый токен: {ttft or 0:.2f} с; весь ответ: {total:.1f} с")
    usage = re.search(r'<div class="usage[^>]*>(.*?)</div>', done_html, re.S)
    if usage:
        print(f"usage: {' '.join(strip_tags(usage.group(1)).split())}")
    answer = strip_tags(re.sub(r'<div class="usage.*', "", done_html, flags=re.S))
    print(f"ответ ({len(''.join(tokens))} симв.):\n  {answer[:600]}")

    if events.get("token", 0) < 2:
        fail("ответ не стримился по токенам")
    if n_sources != EXPECTED_SOURCES:
        fail(f"ожидалось {EXPECTED_SOURCES} источников, получено {n_sources}")
    if "alert-warning" in done_html:
        fail("стрим завершился ошибкой")
    if "[1]" not in answer and "не нашёл" not in answer:
        print("WARN: в ответе нет ссылок [n]")

    r = c.get(stream_url)
    if 'class="answer-body"' not in r.text:
        fail("повторное подключение к стриму не вернуло готовый ответ")
    print("повторное подключение: отдан сохранённый ответ, генерация не перезапускалась")

    api_stream(c, agent_path.rsplit("/", 1)[1], args.question)
    print("OK")


def api_stream(c: httpx.Client, agent_id: str, question: str) -> None:
    """JSON API с Bearer-ключом: SSE-события с JSON в data (ARCHITECTURE §6.3)."""
    r = c.post("/settings/api-keys", data={"name": "smoke"}, headers={"HX-Request": "true"})
    m = re.search(r"(rag_[0-9a-f]{8}_[A-Za-z0-9_-]+)", r.text)
    if not m:
        fail(f"выпуск API-ключа: HTTP {r.status_code}")
    api = httpx.Client(
        base_url=c.base_url,
        headers={"Authorization": f"Bearer {m.group(1)}"},
        timeout=httpx.Timeout(120, connect=5),
    )
    if api.get(f"/api/v1/agents/{agent_id}").status_code != 200:
        fail("API: агент по ключу не читается")
    types: list[str] = []
    with api.stream(
        "POST", f"/api/v1/agents/{agent_id}/query", json={"question": question, "stream": True}
    ) as resp:
        for line in resp.iter_lines():
            if line.startswith("data: "):
                types.append(json.loads(line[6:])["type"])
    if not types or types[-1] != "done" or "token" not in types:
        fail(f"API-стрим: события {types[:5]}…")
    print(f"API: Bearer-ключ, JSON-события {dict((t, types.count(t)) for t in set(types))}")


if __name__ == "__main__":
    main()
