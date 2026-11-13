"""Переключатель языка UI: cookie `lang` и редирект обратно."""

from fastapi import APIRouter
from fastapi.responses import RedirectResponse, Response

from rag_agents.core import i18n
from rag_agents.web.routes.auth import safe_next

router = APIRouter()


@router.get("/lang/{code}")
async def set_language(code: str, next: str = "/") -> Response:
    resp = RedirectResponse(safe_next(next), status_code=303)
    resp.set_cookie(
        i18n.COOKIE, i18n.normalize(code), max_age=365 * 24 * 3600, samesite="lax", httponly=True
    )
    return resp
