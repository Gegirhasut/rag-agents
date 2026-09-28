"""Зависимости FastAPI, общие для HTML-роутов и JSON API: контейнер, кто спрашивает, CSRF."""

from typing import Annotated
from uuid import UUID

import structlog
from fastapi import Depends, HTTPException, Request

from rag_agents.container import Container
from rag_agents.core.security import tokens_equal
from rag_agents.domain.auth import Principal

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
CSRF_HEADER = "X-CSRF-Token"
CSRF_FIELD = "csrf_token"
_FORM_TYPES = ("application/x-www-form-urlencoded", "multipart/form-data")


class NotAuthenticatedError(Exception):
    """Нет сессии или ключа. Web → редирект на /login, API → 401 problem+json."""


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


async def _resolve(request: Request, c: Container) -> Principal | None:
    """Bearer-ключ важнее cookie: так скрипт с ключом не зависит от сессии в браузере."""
    auth = request.headers.get("Authorization")
    if auth:
        scheme, _, token = auth.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return None
        return await c.auth.from_api_key(token)
    session_id = request.cookies.get(c.settings.session_cookie)
    return await c.auth.from_session(session_id) if session_id else None


async def optional_principal(request: Request, c: ContainerDep) -> Principal | None:
    # Кэш на запрос: зависимость используется и роутом, и вложенными зависимостями
    if hasattr(request.state, "principal"):
        cached: Principal | None = request.state.principal
        return cached
    principal = await _resolve(request, c)
    request.state.principal = principal
    if principal is not None:
        structlog.contextvars.bind_contextvars(user_id=str(principal.user_id))
    return principal


async def _csrf_token_from(request: Request) -> str | None:
    header = request.headers.get(CSRF_HEADER)
    if header:
        return header
    # Обычные HTML-формы (без HTMX) присылают токен скрытым полем. Starlette кэширует
    # разобранную форму, поэтому роут прочитает её повторно без второго чтения тела.
    if request.headers.get("content-type", "").startswith(_FORM_TYPES):
        value = (await request.form()).get(CSRF_FIELD)
        return value if isinstance(value, str) else None
    return None


async def current_principal(
    request: Request, principal: Annotated[Principal | None, Depends(optional_principal)]
) -> Principal:
    """Аутентификация + CSRF для cookie-сессии на небезопасных методах (ARCHITECTURE §16).

    Запросы с Bearer-ключом от CSRF освобождены: браузер не подставляет ключ сам.
    """
    if principal is None:
        raise NotAuthenticatedError
    if principal.via == "session" and request.method not in SAFE_METHODS:
        sent = await _csrf_token_from(request)
        if not sent or not principal.csrf_token or not tokens_equal(sent, principal.csrf_token):
            raise HTTPException(403, "CSRF-токен неверный или отсутствует. Обновите страницу.")
    return principal


PrincipalDep = Annotated[Principal, Depends(current_principal)]


def get_owner_id(principal: PrincipalDep) -> UUID:
    return principal.user_id


OwnerDep = Annotated[UUID, Depends(get_owner_id)]


def require_admin(principal: PrincipalDep) -> Principal:
    if not principal.is_admin:
        # 404, а не 403: не подсказываем, что такая страница существует
        raise HTTPException(404, "Страница не найдена")
    return principal


AdminDep = Annotated[Principal, Depends(require_admin)]
