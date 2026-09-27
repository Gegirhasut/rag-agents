"""Справочник технологий для страницы «Под капотом»: что это, аналог в Laravel, где в проекте.

Чистая презентация (тексты и координаты схемы), поэтому живёт в web, а не в domain.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Tech:
    id: str
    title: str
    subtitle: str
    group: str  # client | app | store | queue | ai — цвет узла на схеме
    what: str
    laravel: str
    in_project: list[str]
    code: list[str]
    admin: str | None = None  # ключ AdminUiSpec из services.system
    x: int = 0
    y: int = 0
    notes: list[str] = field(default_factory=list)


NODE_W, NODE_H = 180, 78

NODES: list[Tech] = [
    Tech(
        id="browser",
        title="Браузер",
        subtitle="Bootstrap 5 · HTMX · SSE",
        group="client",
        what=(
            "Страницы рендерит сервер (Jinja2), а HTMX по атрибутам hx-post/hx-get шлёт запросы "
            "и вставляет в страницу готовые HTML-фрагменты. Ответ агента приходит потоком "
            "через Server-Sent Events: браузер держит одно HTTP-соединение (EventSource), "
            "сервер дописывает в него события token/sources/done."
        ),
        laravel="Blade + Livewire/Turbo, только без JS-сборки: вся логика на сервере.",
        in_project=[
            "Форма вопроса → POST, ответ — фрагмент с sse-connect",
            "Прогресс загрузки книг — polling фрагмента каждые 2 с",
        ],
        code=["web/templates/pages/", "web/templates/fragments/answer_stream.html"],
        x=10,
        y=208,
    ),
    Tech(
        id="web",
        title="web",
        subtitle="FastAPI · uvicorn ×2",
        group="app",
        what=(
            "FastAPI — async-фреймворк: роуты — это корутины (async def), зависимости "
            "внедряются через Depends. uvicorn — ASGI-сервер с event loop: один процесс "
            "одновременно обслуживает сотни соединений, пока они ждут БД, Qdrant или LLM "
            "(await не блокирует процесс, в отличие от php-fpm, где воркер занят весь запрос)."
        ),
        laravel=(
            "routes/web.php + контроллеры; Depends ≈ сервис-контейнер; container.py ≈ "
            "ServiceProvider; сервисы (services/) ≈ Actions/Services."
        ),
        in_project=[
            "Принимает загрузки и вопросы, стримит ответ по SSE",
            "Считает эмбеддинг вопроса, ищет в Qdrant, зовёт LLM — всё в одном async-потоке",
        ],
        code=["web/app.py", "web/routes/", "container.py", "services/query.py"],
        x=222,
        y=208,
    ),
    Tech(
        id="postgres",
        title="PostgreSQL 16",
        subtitle="SQLAlchemy async · asyncpg",
        group="store",
        what=(
            "Источник правды: пользователи, агенты, документы, тексты чанков, чаты и сообщения. "
            "SQLAlchemy 2.0 — ORM в стиле Data Mapper с Unit of Work, asyncpg — async-драйвер. "
            "Схема версионируется Alembic."
        ),
        laravel=(
            "Eloquent ≈ SQLAlchemy, но модели не выходят из repositories/ — наружу отдаются "
            "Pydantic-DTO. Alembic ≈ php artisan migrate."
        ),
        in_project=[
            "Статусы документов (queued → processing → done/failed)",
            "Тексты чанков — чтобы переиндексировать без повторного парсинга",
            "История чата, цитаты и usage (токены, тайминги) ответов",
        ],
        code=["models/entities.py", "repositories/", "migrations/", "core/db.py"],
        admin="pgweb",
        x=462,
        y=6,
    ),
    Tech(
        id="redis",
        title="Redis 7",
        subtitle="in-memory KV · pub/sub",
        group="store",
        what=(
            "Хранилище в оперативной памяти. Подходит для часто меняющихся мелких данных "
            "и мгновенной рассылки сообщений (pub/sub)."
        ),
        laravel="Cache/Redis-фасад и broadcasting-драйвер redis.",
        in_project=[
            "ingest:{document_id} — горячий прогресс индексации (hash, TTL 1 ч)",
            "trace:events — канал pub/sub этой страницы, trace:history — последние события",
            "Дальше: кэш ответов, сессии, rate limit",
        ],
        code=["services/progress.py", "services/trace.py"],
        admin="redisinsight",
        x=462,
        y=124,
    ),
    Tech(
        id="rabbitmq",
        title="RabbitMQ 3.13",
        subtitle="AMQP-брокер · DLQ",
        group="queue",
        what=(
            "Брокер сообщений: хранит задачи, пока воркер их не заберёт. Задача → exchange "
            "«tasks» → очередь по routing key. Воркер подтверждает (ack) задачу только после "
            "выполнения. Если воркер упал, задача вернётся в очередь. Необработанная ошибка "
            "уводит сообщение через dead-letter exchange в очередь *.dlq."
        ),
        laravel="Транспорт очереди, как QUEUE_CONNECTION=redis/sqs. Сам ничего не выполняет.",
        in_project=[
            "Очереди ingest.parse, ingest.embed, maintenance, eval + их *.dlq",
            "В сообщении только ID документа, никакого текста",
        ],
        code=["workers/celery_app.py", "docker/rabbitmq.conf"],
        admin="rabbitmq",
        x=462,
        y=282,
    ),
    Tech(
        id="uploads",
        title="volume uploads",
        subtitle="файлы книг на диске",
        group="store",
        what=(
            "Docker volume, примонтирован и в web, и в worker. Web пишет файл потоком "
            "(не держит в памяти целиком), воркер читает его при парсинге."
        ),
        laravel="Storage::disk('local').",
        in_project=["/data/uploads/{agent_id}/{document_id}.txt"],
        code=["core/storage.py"],
        x=462,
        y=404,
    ),
    Tech(
        id="worker",
        title="worker",
        subtitle="Celery 5 · prefork ×2",
        group="app",
        what=(
            "Celery — фреймворк фоновых задач: забирает сообщения из RabbitMQ и выполняет "
            "функции-задачи в процессах-потомках (prefork). Задачи синхронные, поэтому async-код "
            "сервисов гоняется в постоянном event loop процесса (workers/runtime.run). "
            "TransientError → ретрай с экспоненциальной паузой, PermanentError → failed сразу."
        ),
        laravel="php artisan queue:work + Job-классы; Flower ≈ Horizon.",
        in_project=[
            "Индексация книги: парсинг → чанкинг → эмбеддинги батчами → upsert в Qdrant",
            "Токенизатор bge-m3 грузится в родителе до fork: дети делят память (copy-on-write)",
        ],
        code=["workers/tasks/ingest.py", "workers/runtime.py", "services/ingest.py"],
        admin="flower",
        x=700,
        y=350,
    ),
    Tech(
        id="llm",
        title="DeepSeek",
        subtitle="LLM · OpenAI-совм. API",
        group="ai",
        what=(
            "Большая языковая модель в облаке. Получает промпт (персона агента + найденные "
            "чанки + вопрос) и стримит ответ токенами. Отвечает только по переданным "
            "фрагментам, со ссылками [n]."
        ),
        laravel="Внешний HTTP API, как платёжный шлюз: ходим через httpx с таймаутами.",
        in_project=[
            "Стрим токенов пробрасывается в браузер без буферизации",
            "Позже — fallback-цепочка Claude / OpenAI / Ollama (итерация 7)",
        ],
        code=["llm/openai_compat.py", "rag/prompting/templates/answer_v1.txt"],
        x=900,
        y=6,
    ),
    Tech(
        id="qdrant",
        title="Qdrant 1.19",
        subtitle="векторная БД",
        group="ai",
        what=(
            "Хранит «точки»: id + вектор из 1024 чисел + payload (JSON с текстом и метаданными "
            "чанка). Ищет ближайшие по смыслу векторы (косинусная близость) с фильтром по "
            "agent_id. Граф HNSW строится внутри каждого агента (payload_m=16), копия векторов "
            "в int8 держится в RAM, оригиналы float32 лежат на диске."
        ),
        laravel=(
            "Прямого аналога нет. Ближе всего Scout + Meilisearch, но поиск идёт по смыслу, "
            "а не по словам: «смысл жизни» находит «зачем я живу»."
        ),
        in_project=[
            "Одна коллекция на модель эмбеддингов: chunks__bge_m3_567m__1024",
            "Слот под sparse-вектор BM25 уже есть (гибридный поиск — итерация 5)",
        ],
        code=["rag/index/qdrant.py"],
        admin="qdrant",
        x=900,
        y=144,
    ),
    Tech(
        id="ollama",
        title="Ollama · bge-m3",
        subtitle="эмбеддинги · на Windows",
        group="ai",
        what=(
            "Сервер локальных моделей. bge-m3 — нейросеть на 567 млн параметров (веса F16, "
            "~1.2 ГБ): текст на входе, вектор из 1024 чисел на выходе, где похожие по смыслу "
            "тексты дают близкие векторы. Веса модели живут здесь, а Qdrant хранит только её "
            "выходы, векторы чанков."
        ),
        laravel="Внешний сервис по HTTP (POST /api/embed), как Elasticsearch-кластер.",
        in_project=[
            "Индексация: батчи по 32 чанка",
            "Вопрос: один эмбеддинг на запрос (~50–300 мс)",
            "Запущена на Windows-хосте: у CPU виртуалки нет AVX",
        ],
        code=["rag/embeddings/ollama.py"],
        x=900,
        y=282,
    ),
]

# Рёбра схемы (без направления: направление задаёт событие src → dst)
EDGES: list[tuple[str, str]] = [
    ("browser", "web"),
    ("web", "postgres"),
    ("web", "redis"),
    ("web", "rabbitmq"),
    ("web", "uploads"),
    ("web", "llm"),
    ("web", "qdrant"),
    ("web", "ollama"),
    ("rabbitmq", "worker"),
    ("uploads", "worker"),
    ("worker", "postgres"),
    ("worker", "redis"),
    ("worker", "qdrant"),
    ("worker", "ollama"),
]


@dataclass(frozen=True)
class Lib:
    title: str
    what: str
    laravel: str
    code: str


LIBS: list[Lib] = [
    Lib(
        "asyncio",
        "Event loop стандартной библиотеки: await отдаёт управление, пока ждём I/O.",
        "ReactPHP / Swoole / Octane",
        "везде: async def",
    ),
    Lib(
        "Pydantic v2",
        "Валидация и сериализация по аннотациям типов; контракты между слоями.",
        "FormRequest + API Resource/DTO",
        "domain/",
    ),
    Lib(
        "pydantic-settings",
        "Типизированные настройки из окружения и .env.",
        "config/*.php + env()",
        "core/config.py",
    ),
    Lib(
        "SQLAlchemy 2.0",
        "ORM (Data Mapper + Unit of Work) и построитель запросов select().",
        "Eloquent + Query Builder",
        "models/, repositories/",
    ),
    Lib("Alembic", "Миграции схемы, автогенерация по моделям.", "migrations", "migrations/"),
    Lib("Jinja2", "Шаблонизатор HTML.", "Blade", "web/templates/"),
    Lib(
        "tokenizers (HF)",
        "Токенизатор bge-m3 на Rust: режет текст на чанки ≤ 512 токенов.",
        "—",
        "rag/chunking/",
    ),
    Lib(
        "structlog",
        "JSON-логи с контекстом (request_id, document_id).",
        "Log::withContext()",
        "core/logging.py",
    ),
    Lib(
        "httpx",
        "Async HTTP-клиент (Ollama, DeepSeek, RabbitMQ API).",
        "Http-фасад (Guzzle)",
        "rag/embeddings, llm/",
    ),
    Lib("uv", "Менеджер зависимостей, lock-файл и venv.", "Composer", "pyproject.toml, uv.lock"),
    Lib(
        "import-linter",
        "Проверяет правила слоёв: web не импортирует repositories/rag/llm.",
        "deptrac",
        "pyproject.toml",
    ),
]
