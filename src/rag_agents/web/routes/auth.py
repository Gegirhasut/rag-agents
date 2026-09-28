"""Вход, выход и API-ключи в UI (ARCHITECTURE ADR-2, §6.1)."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import ValidationError as PydanticValidationError

from rag_agents.domain.auth import ApiKeyCreate, Principal
from rag_agents.services.auth import InvalidCredentialsError
from rag_agents.services.ratelimit import Rule
from rag_agents.web.deps import ContainerDep, PrincipalDep, optional_principal
from rag_agents.web.templating import templates

router = APIRouter()


def safe_next(target: str | None) -> str:
    """Только относительный путь этого сайта: иначе /login?next=https://evil — open redirect."""
    if not target or not target.startswith("/") or target.startswith("//") or "\\" in target:
        return "/"
    return target


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, c: ContainerDep, next: str = "/") -> Response:
    if await optional_principal(request, c) is not None:
        return RedirectResponse(safe_next(next), status_code=303)
    return templates.TemplateResponse(
        request, "pages/login.html", {"next": safe_next(next), "error": None, "email": ""}
    )


@router.post("/login")
async def login(
    request: Request,
    c: ContainerDep,
    email: Annotated[str, Form()] = "",
    password: Annotated[str, Form()] = "",
    next: Annotated[str, Form()] = "/",
) -> Response:
    st = c.settings
    # Лимит на IP до проверки пароля: argon2 дорогой, перебор не должен грузить CPU
    await c.rate_limiter.check(Rule("login", st.rl_login_per_min, 60), client_ip(request))
    try:
        session_id, _ = await c.auth.login(email, password)
    except InvalidCredentialsError:
        return templates.TemplateResponse(
            request,
            "pages/login.html",
            {"next": safe_next(next), "error": "Неверный email или пароль", "email": email},
            status_code=401,
        )
    response = RedirectResponse(safe_next(next), status_code=303)
    response.set_cookie(
        st.session_cookie,
        session_id,
        max_age=st.session_ttl_s,
        httponly=True,
        samesite="lax",
        secure=st.session_cookie_secure,
        path="/",
    )
    return response


@router.post("/logout")
async def logout(c: ContainerDep, principal: PrincipalDep) -> Response:
    await c.auth.logout(principal)
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(c.settings.session_cookie, path="/")
    return response


def _keys_ctx(principal: Principal, keys: list[object], **extra: object) -> dict[str, object]:
    return {"keys": keys, "email": principal.email, **extra}


@router.get("/settings/api-keys", response_class=HTMLResponse)
async def api_keys_page(request: Request, c: ContainerDep, principal: PrincipalDep) -> Response:
    keys = await c.auth.list_keys(principal.user_id)
    return templates.TemplateResponse(
        request, "pages/api_keys.html", _keys_ctx(principal, list(keys), issued=None, error=None)
    )


@router.post("/settings/api-keys", response_class=HTMLResponse)
async def api_key_create(
    request: Request,
    c: ContainerDep,
    principal: PrincipalDep,
    name: Annotated[str, Form()] = "",
) -> Response:
    issued, error = None, None
    try:
        data = ApiKeyCreate(name=name)
        issued = await c.auth.issue_key(principal.user_id, data.name)
    except PydanticValidationError:
        error = "Название ключа — от 1 до 80 символов"
    keys = await c.auth.list_keys(principal.user_id)
    return templates.TemplateResponse(
        request,
        "fragments/api_keys_panel.html",
        _keys_ctx(principal, list(keys), issued=issued, error=error),
    )


@router.delete("/settings/api-keys/{key_id}", response_class=HTMLResponse)
async def api_key_revoke(
    request: Request, key_id: UUID, c: ContainerDep, principal: PrincipalDep
) -> Response:
    await c.auth.revoke_key(principal.user_id, key_id)
    keys = await c.auth.list_keys(principal.user_id)
    return templates.TemplateResponse(
        request,
        "fragments/api_keys_panel.html",
        _keys_ctx(principal, list(keys), issued=None, error=None),
    )
