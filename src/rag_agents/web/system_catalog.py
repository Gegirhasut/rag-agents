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
        subtitle="HTMX · SSE · или API-клиент",
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
            "Вход по паролю (сессия в cookie) или JSON API /api/v1 с ключом Bearer — "
            "оба канала зовут одни и те же сервисы",
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
            "JSON API /api/v1 (OpenAPI), rate limit на вопросы и загрузки, CSRF для форм",
            "/metrics — Prometheus: HTTP, стадии RAG, токены и $, глубина очередей; "
            "метрики воркеров суммируются из общего volume (multiprocess mode)",
        ],
        code=[
            "web/app.py",
            "web/routes/",
            "api/v1/",
            "container.py",
            "services/query.py",
            "core/metrics.py",
        ],
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
            "Статусы документов (queued → parsing → embedding → done/failed) и счётчик батчей",
            "Тексты чанков — чтобы переиндексировать без повторного парсинга",
            "История чата, цитаты и usage (токены, тайминги) ответов",
            "Пользователи и API-ключи (хранится только sha256 ключа)",
            "eval_runs / eval_items — прогоны качества: метрики каждого вопроса",
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
            "sess:* — сессии входа (TTL продлевается при каждом запросе)",
            "rate limit: sliding window в ZSET, атомарно Lua-скриптом",
            "отметка «лимит Langfuse API исчерпан до …», кэш страницы «Аналитика»",
        ],
        code=[
            "services/progress.py",
            "services/trace.py",
            "services/sessions.py",
            "services/ratelimit.py",
        ],
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
            "Очереди ingest.parse, ingest.embed, maintenance (+ eval про запас) и их *.dlq",
            "В сообщении только ID документа, никакого текста",
            "make dlq-replay возвращает сообщения из DLQ в рабочую очередь",
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
        in_project=["/data/uploads/{agent_id}/{document_id}.<txt|fb2|epub|pdf|docx|zip>"],
        code=["core/storage.py"],
        x=462,
        y=404,
    ),
    Tech(
        id="worker",
        title="workers",
        subtitle="Celery 5 · ingest ×2 · embed ×2 · beat",
        group="app",
        what=(
            "Celery — фреймворк фоновых задач: забирает сообщения из RabbitMQ и выполняет "
            "функции-задачи в процессах-потомках (prefork). Задачи синхронные, поэтому async-код "
            "сервисов гоняется в постоянном event loop процесса (workers/runtime.run). "
            "TransientError → ретрай с экспоненциальной паузой, PermanentError → failed сразу."
        ),
        laravel="php artisan queue:work + Job-классы; Flower ≈ Horizon.",
        in_project=[
            "worker-ingest (очередь ingest.parse): парсинг 5 форматов → структурный чанкинг → "
            "чанки в PG → фан-аут задач эмбеддинга по батчам",
            "worker-embed (ingest.embed): батч → Ollama → upsert в Qdrant; документ становится "
            "done тем батчем, чей атомарный +1 закрыл счётчик",
            "beat раз в минуту запускает sweeper: застрявшие документы переотправляются; "
            "упавшие задачи лежат в *.dlq, make dlq-replay возвращает их",
            "Токенизатор bge-m3 грузится в родителе до fork: дети делят память (copy-on-write)",
        ],
        code=[
            "workers/tasks/ingest.py",
            "workers/runtime.py",
            "services/ingest.py",
            "rag/parsing/",
            "rag/chunking/structural.py",
        ],
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
            "Он же — LLM-судья eval: 3 вызова на вопрос в JSON-режиме, лимит 24 000 токенов "
            "(рассуждения reasoning-модели длинные)",
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
            "agent_id. Граф HNSW строится внутри каждого агента (payload_m=16), векторы лежат "
            "на диске. int8-квантизация выключена: на CPU без AVX Qdrant падал с SIGILL."
        ),
        laravel=(
            "Прямого аналога нет. Ближе всего Scout + Meilisearch, но поиск идёт по смыслу, "
            "а не по словам: «смысл жизни» находит «зачем я живу»."
        ),
        in_project=[
            "Одна коллекция на модель эмбеддингов: chunks__bge_m3_567m__1024",
            "Слот под sparse-вектор BM25 уже есть (гибридный поиск — итерация 5)",
            "eval достаёт 20 кандидатов (hit@20), в промпт идут top_k из настроек агента",
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
    Tech(
        id="eval",
        title="eval",
        subtitle="make eval · профиль tools",
        group="app",
        what=(
            "Одноразовый контейнер из того же образа: прогоняет golden-датасет (JSONL с "
            "эталонными ответами и ожидаемыми главами) через тот же QueryService, что отвечает "
            "пользователю, только без чата. Считает метрики поиска и отказов, а LLM-судья "
            "оценивает ответы. Итог — eval_runs/eval_items, markdown-отчёт, Langfuse Datasets "
            "и страница «Качество»."
        ),
        laravel=(
            "Artisan-команда, которая гоняет feature-тесты на живых данных и пишет отчёт; "
            "парный bootstrap — как A/B-тест с доверительным интервалом."
        ),
        in_project=[
            "59 вопросов: факты, интерпретация, multi-hop, вне корпуса (нужен отказ), инъекции",
            "hit@k, recall@k, MRR — по совпадению найденной главы с ожидаемой",
            "Судья judge_v1: faithfulness, context precision/recall, answer relevancy, "
            "citation support",
            "make eval-diff A=… B=… — разница с 95 % CI: «лучше» только если CI не пересекает 0",
            "Ходит только через services (import-linter), метрики в /metrics не пишет",
        ],
        code=["eval/runner.py", "eval/judge.py", "eval/stats.py", "eval/datasets/", "/eval"],
        x=222,
        y=404,
    ),
    Tech(
        id="langfuse",
        title="Langfuse Cloud",
        subtitle="трейсы · $ · датасеты",
        group="ai",
        what=(
            "Наблюдаемость для LLM: дерево шагов каждого запроса (эмбеддинг, поиск, сборка "
            "промпта, генерация) с токенами, стоимостью и TTFT. SDK на OpenTelemetry копит "
            "span-ы в памяти и отправляет фоновым потоком — ответ сеть не ждёт."
        ),
        laravel="Telescope/Sentry Performance, только для LLM-вызовов и в облаке.",
        in_project=[
            "Трейс на каждый вопрос и индексацию, 👍/👎 → score на трейсе",
            "eval: трейсы прогона в сессии eval-<run>, метрики вопроса — score-ами, "
            "прогон целиком — в Langfuse Datasets",
            "Cloud, а не self-hosted: ClickHouse + MinIO не влезают в 8 ГБ VM",
        ],
        code=["core/observability.py", "core/langfuse_api.py"],
        x=900,
        y=404,
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
    ("web", "langfuse"),
    ("worker", "langfuse"),
    ("eval", "postgres"),
    ("eval", "ollama"),
    ("eval", "qdrant"),
    ("eval", "llm"),
    ("eval", "langfuse"),
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
    Lib(
        "lxml · selectolax",
        "Разбор XML/HTML: fb2 и главы epub (OPF/NAV читаем сами, без AGPL-ebooklib).",
        "DOMDocument / Symfony DomCrawler",
        "rag/parsing/fb2.py, epub.py",
    ),
    Lib(
        "PyMuPDF · python-docx",
        "Текст PDF по страницам (колонтитулы, переносы) и абзацы docx со стилями заголовков. "
        "PyMuPDF — AGPL: допустимо для открытого pet-проекта, запасной вариант pypdfium2.",
        "smalot/pdfparser, PhpWord",
        "rag/parsing/pdf.py, docx.py",
    ),
    Lib(
        "razdel",
        "Деление русского текста на предложения: чанкер не режет посреди фразы.",
        "—",
        "rag/chunking/",
    ),
    Lib(
        "numpy",
        "Векторная математика: PCA карты векторов и парный bootstrap eval (10 000 ресэмплов).",
        "—",
        "services/system.py, eval/stats.py",
    ),
    Lib(
        "prometheus_client",
        "Счётчики и гистограммы /metrics; multiprocess mode суммирует web и воркеры.",
        "promphp/prometheus_client_php",
        "core/metrics.py",
    ),
    Lib(
        "langfuse",
        "SDK трейсов на OpenTelemetry: span-ы копятся и уходят фоновым потоком.",
        "Sentry SDK",
        "core/observability.py",
    ),
    Lib(
        "typer",
        "CLI из аннотаций типов: rag-agents user create, dlq replay, python -m rag_agents.eval.",
        "Artisan-команды",
        "cli.py, eval/__main__.py",
    ),
    Lib("uv", "Менеджер зависимостей, lock-файл и venv.", "Composer", "pyproject.toml, uv.lock"),
    Lib(
        "import-linter",
        "Проверяет правила слоёв: web не импортирует repositories/rag/llm.",
        "deptrac",
        "pyproject.toml",
    ),
]
