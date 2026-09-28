"""Вход в UI для e2e-скриптов: cookie-сессия + CSRF-заголовок на все запросы клиента.

Учётные данные: RAG_EMAIL / RAG_PASSWORD, по умолчанию — seed-администратор
(SEED_USER_EMAIL / SEED_USER_PASSWORD из .env, см. `make seed`).
"""

import os
import re
import sys

import httpx


def dotenv(name: str) -> str | None:
    try:
        with open(".env", encoding="utf-8") as f:
            for line in f:
                key, sep, value = line.strip().partition("=")
                if sep and key == name:
                    return value.strip().strip("'\"") or None
    except FileNotFoundError:
        pass
    return None


def credentials() -> tuple[str, str]:
    email = (
        os.environ.get("RAG_EMAIL")
        or os.environ.get("SEED_USER_EMAIL")
        or dotenv("SEED_USER_EMAIL")
        or "owner@local"
    )
    password = (
        os.environ.get("RAG_PASSWORD")
        or os.environ.get("SEED_USER_PASSWORD")
        or dotenv("SEED_USER_PASSWORD")
    )
    if not password:
        sys.exit(
            "FAIL: нет пароля. Задайте SEED_USER_PASSWORD в .env и выполните `make seed` "
            "(или RAG_EMAIL/RAG_PASSWORD)"
        )
    return email, password


def login(client: httpx.Client) -> str:
    """Логинит клиента и возвращает CSRF-токен (он же проставлен в заголовки клиента)."""
    email, password = credentials()
    r = client.post("/login", data={"email": email, "password": password})
    if r.status_code != 303:
        sys.exit(f"FAIL: вход {email}: HTTP {r.status_code}")
    page = client.get("/")
    m = re.search(r'name="csrf-token" content="([^"]+)"', page.text)
    if not m:
        sys.exit("FAIL: на странице нет csrf-token")
    client.headers["X-CSRF-Token"] = m.group(1)
    return m.group(1)
