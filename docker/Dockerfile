# syntax=docker/dockerfile:1.7
FROM python:3.12.14-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

FROM base AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
# Слой зависимостей кэшируется отдельно от кода
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM base AS runtime
# Токенизатор bge-m3 для подсчёта токенов чанков; ревизия зафиксирована
ARG BGE_M3_REVISION=5617a9f61b028005a4858fdac845db406aefb181
ADD --chmod=644 https://huggingface.co/BAAI/bge-m3/resolve/${BGE_M3_REVISION}/tokenizer.json /opt/models/bge-m3/tokenizer.json
# ADD --chmod применяет режим и к созданным каталогам (без +x) — возвращаем обход каталогов
RUN chmod -R a+rX /opt/models \
    && useradd --system --uid 10001 app && mkdir -p /data/uploads && chown -R app /data
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY alembic.ini ./
COPY migrations ./migrations
COPY configs ./configs
ENV PATH=/app/.venv/bin:$PATH
USER app
EXPOSE 8000
CMD ["uvicorn", "rag_agents.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
