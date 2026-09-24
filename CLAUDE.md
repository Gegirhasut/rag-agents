# CLAUDE.md

RAG-агенты по корпусу документов: пользователь создаёт агента, загружает книги (txt, fb2, epub, pdf, docx) и получает ответы только по ним, со ссылками на источники и стримингом. Pet-проект и портфолио (Senior Python AI Developer).

**Документы (читать перед задачей):** [docs/SPEC.md](docs/SPEC.md) — требования и допущения; [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — решения, контракты, схема БД, RAG-пайплайн; [docs/PLAN.md](docs/PLAN.md) — итерации. Если реализация расходится с ARCHITECTURE, обнови документ **в том же изменении** (или раздел ADR с причиной).

**Автор:** senior PHP/Laravel (20+ лет), растёт в Python AI-стеке. Объясняй решения и Python-специфику (asyncio, типизация, Pydantic, SQLAlchemy 2.0), азы программирования не объясняй. Параллели с Laravel уместны.

**Статус:** этап проектирования завершён, кода пока нет. Команды ниже — целевые, появятся в итерации 1.

## Стек
- Python 3.12, **uv** (зависимости, lock, venv), один пакет `rag_agents` (src-layout) с extras (`rerank`, `eval`, `admin`).
- FastAPI + Jinja2 + HTMX (+ `htmx-ext-sse`) + Bootstrap 5, серверный рендеринг, **без SPA и без node-сборки** (статика с CDN или вендорится в `static/vendor`).
- Pydantic v2, pydantic-settings, SQLAlchemy 2.0 async + asyncpg, Alembic.
- Celery 5 + RabbitMQ (брокер, DLQ), Redis (кэш, сессии, rate limit, result backend).
- PostgreSQL 16, Qdrant (dense + sparse, гибрид через Query API и RRF).
- Ollama `bge-m3` (эмбеддинги, CPU), fastembed BM25 (russian), reranker `bge-reranker-v2-m3` int8 ONNX (отдельный сервис, onnxruntime без torch).
- LLM: DeepSeek (основной, OpenAI-совместимый) → fallback Anthropic Claude / OpenAI / Ollama через единый `LLMRouter`.
- Langfuse Cloud (трейсы), structlog (JSON-логи), prometheus_client (`/metrics`).
- RAG-ядро своё (ADR-3). LlamaIndex — только как эталон для сравнения в eval (`eval/baselines/`, итерация 6), LangChain не используем.
- **Лицензии моделей и библиотек — только коммерчески допустимые** (Apache 2.0 / MIT / BSD). NC-лицензии (например, jina-reranker-v2) запрещены. Лицензию новой модели проверять по карточке до добавления.

## Структура монорепо (целевая)
```
rag-agents/
├── CLAUDE.md  README.md  pyproject.toml  uv.lock  Makefile  compose.yaml  .env.example
├── docker/                    # Dockerfile (multi-stage, один образ; команды задаёт compose)
├── docs/                      # SPEC, ARCHITECTURE, PLAN
├── migrations/                # Alembic (async env)
├── src/rag_agents/
│   ├── core/                  # config (Settings), db (engine, UoW), redis, logging, security, errors
│   ├── domain/                # Pydantic-схемы и enums: контракты между слоями (без логики I/O)
│   ├── models/                # SQLAlchemy ORM-модели (только для repositories)
│   ├── repositories/          # доступ к PG; возвращают domain-DTO, не ORM
│   ├── services/              # use cases: agents, documents, ingest, query, chat, auth, eval
│   ├── rag/
│   │   ├── parsing/           # txt, fb2, epub, pdf, docx → Iterator[Section]
│   │   ├── cleaning/          # нормализация, PDF-артефакты, орфография
│   │   ├── chunking/          # структурный чанкер, токенизатор bge-m3
│   │   ├── embeddings/        # ollama (dense), bm25 (sparse)
│   │   ├── index/             # QdrantChunkIndex (agent_id обязателен)
│   │   ├── retrieval/         # hybrid, rerank_client, context builder
│   │   └── prompting/         # шаблоны промптов (версионируются), цитаты
│   ├── llm/                   # LLMProvider protocol, openai_compat, anthropic, router, breaker, prices
│   ├── web/                   # HTML-роуты, templates/, static/, SSE-рендер
│   ├── api/v1/                # JSON API
│   ├── workers/               # celery_app, queues, runtime (loop per process), tasks/
│   ├── reranker/              # отдельная точка входа сервиса rerank
│   ├── eval/                  # runner, метрики, калибровка, отчёты; baselines/llamaindex.py
│   └── cli.py                 # typer: user create, dlq replay, reindex
├── eval/datasets/             # golden JSONL (в git)
├── configs/eval/              # конфигурации экспериментов
├── configs/rag/thresholds.yaml  # откалиброванные пороги отказа по (reranker, embedding)
├── reports/eval/              # сгенерированные отчёты (в git — только избранные)
└── tests/{unit, integration, e2e, fixtures}
```

## Правила архитектуры
- Слои: `web`/`api` → `services` → `repositories` | `rag` | `llm`. Роуты не ходят в репозитории, Qdrant и LLM напрямую. `workers` только вызывают сервисы. Проверяется `import-linter`.
- **Изоляция агентов не обсуждается:** каждый метод репозитория и индекса, который трогает данные агента, принимает `agent_id` обязательным аргументом. Сервис проверяет владельца. На чужое — 404. Новые эндпоинты добавляются в `tests/integration/test_isolation.py`.
- ORM-модели не выходят за `repositories`. Наружу — Pydantic-DTO из `domain`.
- Транзакцию коммитит сервис (UoW). Публикация Celery-задач — только после коммита (`uow.on_commit`).
- В Celery-сообщениях только ID (Pydantic-модели из `domain/tasks.py`), никакого контента.
- Задачи идемпотентны: claim через условный `UPDATE … RETURNING`, детерминированные uuid5 для точек Qdrant. Ошибки делятся на `TransientError` (ретрай) и `PermanentError` (сразу `failed`), остальное уходит в DLQ.
- Всё I/O — async. Celery-задачи — sync-обёртки над корутинами через `workers/runtime.run()`: один event loop на процесс. `asyncio.run()` в задачах не использовать.
- Промпты — в `rag/prompting/templates/` с версией (`prompt_version`). Изменение промпта = новая версия + eval-прогон.
- Порог отказа — `RetrievalSettings.min_rerank_score` (None = дефолт из `configs/rag/thresholds.yaml`). Хардкодить порог в коде нельзя. Смена модели reranker-а или эмбеддингов → `make calibrate`.
- Имена моделей LLM и эмбеддингов, URL и ключи — только из `Settings`/`.env`, не хардкодить.
- Любое изменение retrieval, чанкинга или промпта сопровождается eval-прогоном (`make eval`), а результат упоминается в описании коммита или PR.

## Правила кода
- **ruff** (lint + format), line-length 100, правила: `E,F,W,I,UP,B,SIM,ASYNC,S,RUF,PL,PT,TRY,DTZ` (конфиг в `pyproject.toml`). Никаких `# noqa` без комментария с причиной.
- **mypy --strict** для `src/` (плагин pydantic). `Any` — только на границах с внешними SDK, с комментарием.
- Типы: `X | None`, `list[str]`, `Annotated`, `Protocol` для интерфейсов (`Parser`, `Embedder`, `Reranker`, `LLMProvider`). Даты — только tz-aware (`datetime.now(UTC)`).
- Pydantic v2 API (`model_validate`, `model_dump`, `ConfigDict`), без v1-совместимости.
- SQLAlchemy 2.0 style (`select()`, `Mapped[...]`, `mapped_column`), без legacy `Query`.
- Логи через `structlog.get_logger()`, с ключами, а не f-строками: `log.info("ingest.batch_done", batch_no=n)`. Тексты пользователей и документов в логи не писать.
- Docstring — для публичных функций сервисов и `rag/`, кратко, по делу. Комментарии объясняют «почему», а не «что».
- Шаблоны: HTMX-фрагменты лежат в `templates/fragments/`, страницы — в `templates/pages/`. Никакой бизнес-логики в Jinja.

## Тесты
- pytest + pytest-asyncio (`asyncio_mode = "auto"`), `respx` для HTTP-моков LLM и Ollama, фабрики — `polyfactory`.
- `tests/unit` — без сети и БД (парсеры на фикстурах, чанкер, промпт, цитаты, роутер LLM на записанных стримах).
- `tests/integration` — реальные PG, Redis, Qdrant из `compose.test.yaml` (tmpfs, отдельные порты `15432/16379/16333`), схема через Alembic. LLM и Ollama замоканы.
- `tests/e2e` — smoke по поднятому стенду (помечены `@pytest.mark.e2e`, в CI не запускаются).
- Покрытие ядра (`rag/`, `llm/`, `services/`) ≥ 80 %.
- Баг-фикс начинается с падающего теста.

## Команды (целевые, `Makefile`)
```bash
make up            # docker compose up -d (профиль по умолчанию)
make up-debug      # + flower
make down          # docker compose down
make logs s=web    # логи сервиса
make migrate       # alembic upgrade head (в контейнере migrate)
make revision m="add chats"   # alembic revision --autogenerate
make seed          # admin-пользователь + демо-агент
make sh            # shell в контейнере web

make lint          # ruff check . && ruff format --check . && mypy src && lint-imports
make fmt           # ruff format . && ruff check --fix .
make test          # pytest tests/unit tests/integration (поднимает compose.test.yaml)
make test-unit     # только unit, без docker

make eval AGENT=tolstoy CONFIG=hybrid_rerank   # eval-прогон → reports/eval/
make calibrate AGENT=tolstoy                    # калибровка порога отказа → configs/rag/thresholds.yaml
make compare AGENT=tolstoy                      # своё ядро vs LlamaIndex → reports/eval/*_core_vs_llamaindex.md
make dlq-replay QUEUE=ingest.embed
```
Локально без docker: `uv sync --all-extras`, `uv run pytest tests/unit`, `uv run uvicorn rag_agents.web.app:app --reload`. Python 3.12 ставится через `uv python install 3.12` (системный на VM — 3.10).

## Окружение (VM)
- Ubuntu 22.04, 6 vCPU, 7.8 GiB RAM + 2 GiB swap, без GPU. Docker 29 + Compose v5. Логи json-file 10m × 3.
- UI с Windows-хоста: `http://192.168.56.10:8080`. Остальные порты — только `127.0.0.1` (Docker обходит ufw). Таблица портов и лимитов памяти — в ARCHITECTURE §13. **Суммарный бюджет ~6.6 GiB, новые сервисы только с пересчётом.**
- Ollama для LLM (если нужен локальный) — на Windows-хосте `192.168.56.1:11434`. На VM память есть только под bge-m3.
- На VM лежит код проекта ботов (`~/code/bots.ai`) — не трогать.

## Git
- Ветка по умолчанию `main`. Фичи — в ветках `iter-N/<кратко>`.
- Коммиты — Conventional Commits (`feat(ingest): …`, `fix(rag): …`, `docs: …`), на английском, тело — по желанию.
- `.env`, `data/`, `reports/eval/*` (кроме избранных) — в `.gitignore`.
