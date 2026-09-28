"""Ошибки → HTTP. Одни и те же исключения сервисов отдаются по-разному:
JSON API — RFC 9457 problem+json, HTMX — фрагмент-алерт, обычная навигация — страница."""

from http import HTTPStatus
from typing import Any
from urllib.parse import quote, urlsplit

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from markupsafe import escape
from starlette.exceptions import HTTPException as StarletteHTTPException

from rag_agents.services.errors import ConflictError, NotFoundError, ValidationError
from rag_agents.services.ratelimit import RateLimitedError
from rag_agents.web.deps import NotAuthenticatedError
from rag_agents.web.templating import templates

API_PREFIX = "/api/"
PROBLEM_JSON = "application/problem+json"

_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    422: "validation_error",
    429: "rate_limited",
}


def is_api(request: Request) -> bool:
    return request.url.path.startswith(API_PREFIX)


def problem(
    status: int,
    detail: str,
    *,
    code: str | None = None,
    headers: dict[str, str] | None = None,
    **extra: Any,
) -> JSONResponse:
    code = code or _CODES.get(status, "error")
    body = {
        "type": f"/problems/{code}",
        "title": HTTPStatus(status).phrase,
        "status": status,
        "detail": detail,
        "code": code,
        **extra,
    }
    return JSONResponse(body, status_code=status, media_type=PROBLEM_JSON, headers=headers)


def _html_error(
    request: Request, status: int, detail: str, headers: dict[str, str] | None = None
) -> Response:
    if request.headers.get("HX-Request"):
        return HTMLResponse(
            f'<div class="alert alert-danger py-2 my-2">{escape(detail)}</div>',
            status_code=status,
            headers=headers,
        )
    return templates.TemplateResponse(
        request,
        "pages/error.html",
        {"status": status, "detail": detail},
        status_code=status,
        headers=headers,
    )


def _error(
    request: Request, status: int, detail: str, headers: dict[str, str] | None = None
) -> Response:
    if is_api(request):
        return problem(status, detail, headers=headers)
    return _html_error(request, status, detail, headers)


def _login_url(request: Request) -> str:
    # Для HTMX возвращаемся на страницу, с которой шёл запрос, а не на URL фрагмента
    current = request.headers.get("HX-Current-URL")
    if current:
        parts = urlsplit(current)
        target = parts.path + (f"?{parts.query}" if parts.query else "")
    else:
        target = request.url.path + (f"?{request.url.query}" if request.url.query else "")
    return f"/login?next={quote(target, safe='/')}" if target not in ("", "/") else "/login"


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(NotAuthenticatedError)
    async def not_authenticated(request: Request, exc: NotAuthenticatedError) -> Response:
        if is_api(request):
            return problem(
                401,
                "Нужен заголовок Authorization: Bearer rag_… или вход через /login",
                headers={"WWW-Authenticate": "Bearer"},
            )
        url = _login_url(request)
        if request.headers.get("HX-Request"):
            # HTMX не следует 30x для навигации: просим клиент перейти сам
            return Response(status_code=401, headers={"HX-Redirect": url})
        return RedirectResponse(url, status_code=303)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
        return _error(
            request, exc.status_code, str(exc.detail), dict(exc.headers) if exc.headers else None
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> Response:
        if is_api(request):
            errors = [
                {"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type")}
                for e in exc.errors()
            ]
            return problem(422, "Некорректный запрос", errors=errors)
        return _html_error(request, 422, "Некорректные данные формы")

    @app.exception_handler(NotFoundError)
    async def not_found(request: Request, exc: NotFoundError) -> Response:
        return _error(request, 404, "Не найдено")

    @app.exception_handler(ConflictError)
    async def conflict(request: Request, exc: ConflictError) -> Response:
        return _error(request, 409, str(exc))

    @app.exception_handler(ValidationError)
    async def invalid(request: Request, exc: ValidationError) -> Response:
        return _error(request, 422, str(exc))

    @app.exception_handler(RateLimitedError)
    async def rate_limited(request: Request, exc: RateLimitedError) -> Response:
        headers = {"Retry-After": str(exc.retry_after_s)}
        detail = f"Слишком много запросов. Повторите через {exc.retry_after_s} с."
        if is_api(request):
            return problem(429, detail, headers=headers, retry_after=exc.retry_after_s)
        return _html_error(request, 429, detail, headers)
