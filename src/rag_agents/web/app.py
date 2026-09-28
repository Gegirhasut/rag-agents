from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import Depends, FastAPI
from fastapi.security import HTTPBearer
from fastapi.staticfiles import StaticFiles

import rag_agents.api.v1 as api_v1
from rag_agents.container import Container, build_container
from rag_agents.core.config import Settings, get_settings
from rag_agents.core.logging import configure_logging
from rag_agents.web.errors import install_error_handlers
from rag_agents.web.middleware import RequestContextMiddleware
from rag_agents.web.routes import auth, chat, documents, health, insights, pages, system
from rag_agents.web.templating import templates

log = structlog.get_logger()
STATIC_DIR = Path(__file__).parent / "static"


ContainerFactory = Callable[[Settings], Container]
# Схема Bearer только для OpenAPI (кнопка Authorize в /api/docs); проверяет ключ current_principal
_bearer_doc = HTTPBearer(auto_error=False, description="API-ключ rag_…, выпуск: /settings/api-keys")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    factory: ContainerFactory = app.state.container_factory
    container = factory(settings)
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


def create_app(container_factory: ContainerFactory = build_container) -> FastAPI:
    """container_factory подменяется в integration-тестах (публикатор задач без RabbitMQ)."""
    app = FastAPI(
        title="RAG Agents API",
        version="1",
        description="Агенты, документы и вопросы по корпусу. Ошибки — application/problem+json.",
        lifespan=lifespan,
        openapi_url="/api/v1/openapi.json",
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.state.container_factory = container_factory
    app.add_middleware(RequestContextMiddleware)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    # В OpenAPI попадает только JSON API; HTML-роуты — детали UI
    for web_router in (
        health.router,
        auth.router,
        pages.router,
        documents.router,
        chat.router,
        system.router,
        insights.router,
    ):
        app.include_router(web_router, include_in_schema=web_router is health.router)
    app.include_router(api_v1.router, dependencies=[Depends(_bearer_doc)])
    install_error_handlers(app)
    return app


app = create_app()
