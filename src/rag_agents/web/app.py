from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from rag_agents.container import build_container
from rag_agents.core.config import get_settings
from rag_agents.core.logging import configure_logging
from rag_agents.web.routes import chat, documents, health, insights, pages, system
from rag_agents.web.templating import templates

log = structlog.get_logger()
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    container = build_container(settings)
    # Итерация 1: без авторизации, все агенты принадлежат seed-пользователю (auth — итерация 2)
    app.state.owner_id = await container.agents.ensure_user(settings.seed_user_email)
    app.state.container = container
    # Ссылки «трейс» в шаблонах показываем, только если трейсинг включён
    templates.env.globals["insights_enabled"] = container.insights.enabled
    log.info(
        "web.started",
        llm_model=settings.llm_model,
        effort=settings.llm_reasoning_effort,
        langfuse=container.tracer.enabled,
    )
    yield
    await container.aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Agents", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(health.router)
    app.include_router(pages.router)
    app.include_router(documents.router)
    app.include_router(chat.router)
    app.include_router(system.router)
    app.include_router(insights.router)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> HTMLResponse:
        if request.headers.get("HX-Request"):
            return HTMLResponse(
                f'<div class="alert alert-danger py-2 my-2">{exc.detail}</div>',
                status_code=exc.status_code,
            )
        return templates.TemplateResponse(
            request,
            "pages/error.html",
            {"status": exc.status_code, "detail": exc.detail},
            status_code=exc.status_code,
        )

    return app


app = create_app()
