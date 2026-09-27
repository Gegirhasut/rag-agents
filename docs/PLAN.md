# PLAN — итерации

> 10 итераций. MVP — итерации 1–8 (критерии приёмки в [SPEC §8](SPEC.md#8-критерии-приёмки-mvp-итерации-18)), 9 — hardening, 10 — «позже» из ТЗ.
> Каждая итерация заканчивается **демо** (что показать за 3 минуты), зелёным CI и обновлёнными docs.
> Оценки даны в «фокус-днях» (ориентир для pet-проекта, не обязательство).

Порядок итераций выбран так:
- **сквозной путь** появляется в первый же день: любое следующее улучшение видно в UI;
- **eval** появляется до тюнинга retrieval (итерация 4 раньше 5): каждое улучшение подтверждается цифрой, а не ощущением;
- **сравнение с LlamaIndex** (итерация 6) идёт после тюнинга своего ядра: сравниваем лучшее с лучшим, а не черновик с фреймворком.

---

## Итерация 1 — Сквозной скелет (3–4 дня)

**Цель.** Один проход «UI → загрузка txt → очередь → Qdrant → вопрос → стрим DeepSeek» работает в Docker Compose.

**Задачи**
1. Монорепо: `pyproject.toml` (uv, Python 3.12), ruff, mypy, pytest, `Makefile`, `.env.example`, `compose.yaml`, `Dockerfile` (multi-stage, uv).
2. Compose: postgres, redis, rabbitmq, qdrant, ollama (+ одноразовый `ollama-pull bge-m3`), `migrate`, web, worker (пока **один** воркер на все очереди). Лимиты памяти из ARCHITECTURE §13.
3. `core/`: `Settings` (pydantic-settings), async engine и session, логирование (structlog, минимально).
4. Alembic + первая миграция: `users` (одна запись seed-пользователя), `agents`, `agent_indexes`, `documents`, `chunks`. Поле `owner_id` есть с первого дня, даже без auth.
5. Web (Jinja2 + HTMX + Bootstrap через CDN): список агентов, создание агента, страница агента, загрузка одного `.txt`, таблица файлов (статус обновляется по F5 или простому polling).
6. Celery app + задача `ingest_document` (одна задача без фан-аута): чтение txt (utf-8 / cp1251), наивный чанкинг по абзацам ~400 токенов, эмбеддинги Ollama, upsert в Qdrant (коллекция `chunks__bge_m3_567m__1024`, **сразу** tenant-индекс по `agent_id`).
7. `llm/`: `OpenAICompatProvider` для DeepSeek (stream).
8. Query: dense-поиск top-6 с фильтром агента → простой промпт с правилом «только по источникам» → SSE `token`/`done` через `htmx-ext-sse`. Список источников под ответом.
9. Тесты: unit на чанкер, integration на репозиторий Qdrant (фильтр агента), smoke e2e (скрипт `make smoke`).

**Структура модулей (появляется в итерации)**
```
src/rag_agents/{core, domain, web, services, repositories, rag/{parsing/txt.py, chunking/naive.py, embeddings/ollama.py, index/qdrant.py}, llm/{base.py, openai_compat.py}, workers/{celery_app.py, tasks/ingest.py}}
migrations/  templates/  static/  tests/
```

**Критерии готовности / демо**
- `make up && make migrate && make seed` поднимает стенд, `http://192.168.56.10:8080` открывается с Windows.
- Создаю агента «Толстой (тест)», загружаю «Исповедь» в txt → через ≤ 5 мин статус `done`.
- Спрашиваю «В чём смысл жизни?» → ответ идёт потоком, под ним 6 фрагментов из «Исповеди».
- Второй агент с другим текстом не видит чанки первого (integration-тест).
- `make lint test` зелёные.

---

## Мини-итерация 1.5 — Langfuse Cloud (сделано 2026-09-27)

Вынесено из итерации 4 (задача 1): от auth и ingest-очередей не зависит, а трейсы нужны уже сейчас для разбора ответов. Детали — ARCHITECTURE ADR-9 и §14.2.
- Трейс на вопрос: `embed_query` → `qdrant_search` → `llm_generate` (токены, TTFT, reasoning effort, cost по model definition); session = чат, user = владелец, tags = агент.
- Трейс ingest: `parse` → `chunk` → `save_chunks` → `embed_upsert` (батчи) → `finalize`, без текстов.
- 👍/👎 под ответом → `messages.feedback` + score `user_feedback` (миграция `0002`).
- No-op без ключей и в тестах; flush при остановке web и воркера. Проверка: `make langfuse-check`, `make langfuse-model`, `make langfuse-trace id=…`.
- Страница «Аналитика» `/insights` (ARCHITECTURE §14.5): KPI, график, агенты, шаги пайплайна, последние трейсы с водопадом span-ов, сессии чатов и документов — данные Langfuse API у нас в UI. Демо-данные: `make demo-traffic`.

**Долг перед итерацией 3:** когда ingest разойдётся на задачи parse и embed, трейс собирается по детерминированному `trace_id` от `document_id`.

---

## Итерация 2 — Фундамент: auth, API, статусы, CI (3 дня)

**Цель.** Проект выглядит как продукт: вход по паролю, JSON API с ключами, живые статусы, CI.

**Задачи**
1. Auth (ADR-2): `users`, argon2id, серверные сессии в Redis, CSRF-middleware для HTMX, CLI `rag-agents user create --admin`.
2. API-ключи: таблица `api_keys`, выпуск и отзыв в UI, зависимость `current_principal` (сессия | bearer).
3. JSON API `/api/v1`: agents CRUD, documents (upload, list, get, delete), `query` (обычный + SSE). Ошибки в формате problem+json. OpenAPI-документация.
4. Статусы: `status/stage/progress`, Redis hash прогресса, HTMX polling каждые 2 с с остановкой по **HTTP 286**.
5. Проверка владельца во всех сервисах + `tests/integration/test_isolation.py` (параметризованный обход эндпоинтов).
6. Rate limit (Redis, sliding window) на вопросы и логин.
7. structlog с `request_id`, `/healthz`, `/readyz`.
8. GitHub Actions: ruff, mypy, pytest (сервисы PG, Redis, Qdrant в services-контейнерах), `import-linter`.

**Критерии готовности / демо**
- Логин и логаут; чужой агент по прямой ссылке → 404.
- `curl -N -H "Authorization: Bearer rag_…" .../api/v1/agents/{id}/query -d '{"question":"…","stream":true}'` стримит JSON-события.
- Загружаю 3 файла → строки сами меняют статусы, polling останавливается, когда всё `done` (видно в DevTools).
- Бейдж CI в README зелёный.

---

## Итерация 3 — Ingest production-grade: форматы, структура, очереди (5–6 дней)

**Цель.** Любая книга в 5 форматах до 100 МБ надёжно превращается в структурированные чанки. Сбои не теряют документы.

**Задачи**
1. Парсеры (генераторы секций): txt (кодировки, эвристика глав), fb2 / fb2.zip, epub (spine + TOC), pdf (PyMuPDF, outline, постранично, детект скана), docx (стили заголовков). Фикстуры: маленькие реальные файлы каждого формата в `tests/fixtures/`.
2. Очистка (§7.3): PDF-переносы, колонтитулы, `norm_text` с ё и дореформенной орфографией.
3. Структурный чанкер (§7.4): границы секций, `razdel`, target/max/min/overlap, contextual header, метаданные. Токенизатор bge-m3.
4. Топология RabbitMQ (§10): очереди `ingest.parse`, `ingest.embed`, `maintenance`, DLX/DLQ; два воркера (`worker-ingest`, `worker-embed`) + `beat`.
5. Фан-аут батчей, `ingest_jobs`, `embed_batches`, claim-UPDATE, атомарная финализация.
6. Классы ошибок Transient / Permanent, backoff + jitter, reject → DLQ; sweeper в beat; `make dlq-replay`.
7. Дедупликация по sha256, zip-bomb и XXE-защита, лимиты.
8. Страница «Документ»: оглавление + просмотр чанков (отладка парсинга).
9. Удаление документа и агента (асинхронная очистка PG, Qdrant, файлов), `corpus_version`.
10. **Замеры**: время ingest «Войны и мира» (fb2) и 50 МБ PDF, пиковая память воркеров (`docker stats`). Записать в SPEC §9.

**Структура модулей**
```
rag/parsing/{base.py, txt.py, fb2.py, epub.py, pdf.py, docx.py, headings.py, encoding.py}
rag/cleaning/{normalize.py, pdf_artifacts.py, orthography.py}
rag/chunking/{structural.py, tokenizer.py}
workers/tasks/{parse.py, embed.py, maintenance.py, sweeper.py}; workers/{queues.py, errors.py, runtime.py (event loop per process)}
```

**Критерии готовности / демо**
- Агент «Внутренний мир Льва Толстого»: «Исповедь» (txt cp1251), «Война и мир» (fb2), «Анна Каренина» (epub), дневники (pdf), статья (docx) — все `done`, главы видны в оглавлении.
- **Chaos-демо:** `docker kill worker-embed` посреди «Войны и мира» → после рестарта документ доезжает до `done`, число точек в Qdrant = `chunks_total` (без дублей).
- Битый PDF → `failed: Файл повреждён`, без ретраев. Искусственное исключение → сообщение в `ingest.embed.dlq` → replay → `done`.
- Замеры ingest и памяти записаны.

---

## Итерация 4 — Eval-харнесс и наблюдаемость: baseline в цифрах (3–4 дня)

**Цель.** До любого тюнинга есть воспроизводимая оценка качества и трейсы каждого запроса.

**Задачи**
1. ~~Langfuse Cloud: SDK, трейсы query и ingest, trace_id в `messages`, no-op без ключей~~ — **сделано раньше, в мини-итерации 1.5** (2026-09-27, ARCHITECTURE ADR-9 и §14.2), плюс 👍/👎 → score. В итерации 4 остаётся: span-ы rerank и build_context, выгрузка eval в Langfuse Datasets.
2. Golden-датасет `eval/datasets/tolstoy.jsonl`: 40–60 вопросов, 4 категории (ARCHITECTURE §15.1), включая `out_of_corpus` (≥ 20 %, это нужно для калибровки порога) и `injection`.
3. Eval runner (`python -m rag_agents.eval`): вызывает `QueryService` напрямую, считает retrieval-метрики (hit@k, recall@k, MRR), RAGAS (faithfulness, answer relevancy, context precision и recall) с judge DeepSeek, метрики отказов и цитат, латентности, $. Парный bootstrap для сравнения двух прогонов.
4. Runner сохраняет `max_rerank_score` / `max_dense_score` каждого вопроса: это сырьё для калибровки в итерации 5.
5. `eval_runs` / `eval_items`, markdown-отчёт в `reports/eval/`, выгрузка в Langfuse Datasets.
6. Baseline своего ядра: конфиг `dense`.
7. Метрики Prometheus (ARCHITECTURE §14.3) на `/metrics`.
8. CI: мини-eval retrieval на фикстурном корпусе с замоканным LLM.

**Критерии готовности / демо**
- `make eval AGENT=tolstoy CONFIG=dense` → отчёт с таблицей метрик и 10 худшими примерами.
- `make eval-diff A=<run> B=<run>` печатает разницы с 95 % CI.
- В Langfuse видно дерево спанов запроса с токенами и стоимостью.

---

## Итерация 5 — Качество retrieval и ответа: гибрид, rerank, порог отказа, цитаты (5 дней)

**Цель.** Лучшее качество, которое даёт CPU. Каждое улучшение подтверждено eval-ом, порог отказа откалиброван.

**Задачи**
1. Sparse BM25 (fastembed, russian) в ingest и query. Переиндексация демо-агента. Qdrant Query API: prefetch dense + sparse + RRF, фильтр агента в каждом prefetch.
2. **Reranker `BAAI/bge-reranker-v2-m3` (Apache 2.0)** (ARCHITECTURE ADR-8, §7.7):
   - одноразовый сервис `reranker-export`: `optimum` → ONNX → dynamic int8, в volume `models`;
   - сервис `reranker` на `onnxruntime` + `tokenizers` без torch;
   - клиент с таймаутом и деградацией до RRF;
   - замер латентности на 8 / 16 / 24 кандидатах;
   - eval-сравнение с `mmarco-mMiniLMv2` (лицензию проверить по карточке) по качеству и p95. Выбор модели и `rerank_candidates` фиксируется в ADR-8.
3. Сборка контекста (ARCHITECTURE §7.8): small-to-big, склейка, бюджет, группировка.
4. Промпт v3 (§7.9) с персоной и отказом **до LLM** по порогу.
5. **Порог отказа как настройка агента:** `min_rerank_score` / `min_dense_score` в `RetrievalSettings` (`None` = глобальный дефолт), ползунок «Строгость» и кнопка «Сбросить к дефолту» в настройках агента, эффективный порог в трейсе и в `messages.usage`.
6. **Калибровка порога на golden-датасете** (ARCHITECTURE §15.4) — отдельная задача с артефактом:
   - `make calibrate AGENT=tolstoy`: сетка τ, F0.5 по `out_of_corpus`, кривая precision/recall;
   - проверка leave-one-document-out;
   - запись в `configs/rag/thresholds.yaml` с ключом `(reranker, embedding)` и `dataset_sha`;
   - CI-проверка, что для текущей пары моделей есть запись.
7. Пост-проверка цитат, `grounded`, markdown → sanitized HTML, поповеры `[n]`, событие `sources` до генерации, «Также найдено».
8. Eval-серия: `dense` → `hybrid` → `hybrid_rerank` → `+neighbors` → размеры чанка 256 / 400 / 800. Выбор дефолтов по результатам (с CI), таблица фиксируется в SPEC §9. Лучшая конфигурация получает имя `core-best`.

**Структура модулей**
```
rag/embeddings/bm25.py; rag/retrieval/{hybrid.py, rerank_client.py, context.py, thresholds.py}; rag/prompting/{templates/, builder.py, citations.py}
reranker/{app.py, export.py}; eval/calibrate.py; configs/rag/thresholds.yaml
```

**Критерии готовности / демо**
- Таблица «dense / hybrid / hybrid+rerank» с recall@8, MRR, faithfulness, refusal F1, p95 латентности и CI разниц.
- **Отчёт калибровки:** кривая P/R по τ, выбранный τ, разброс по leave-one-document-out. `thresholds.yaml` закоммичен.
- «Какой курс биткоина?» → отказ, в трейсе нет LLM-спана и виден эффективный порог. Если выкрутить «Строгость» агента в 0, тот же вопрос дойдёт до LLM, и она откажется по правилу промпта.
- Клик по `[2]` → поповер «Л. Н. Толстой — „Исповедь“, гл. IV» с текстом фрагмента.
- Замер rerank: p95 на 16 кандидатах ≤ 2.5 с (или задокументированное решение перейти на запасную модель).
- `docker stop reranker` → ответы продолжают работать (деградация, порог `min_dense_score`), метрика `rerank_skipped` растёт.

---

## Итерация 6 — Своё ядро vs LlamaIndex на одном датасете (3 дня, обязательная)

**Цель.** Проверить ADR-3 цифрами: даёт ли своё ядро выигрыш по сравнению с LlamaIndex при прочих равных, и какой ценой.

**Задачи** (протокол — ARCHITECTURE §15.3)
1. `src/rag_agents/eval/baselines/llamaindex.py` (extra `eval`):
   - `LI-default` — `SentenceSplitter` по умолчанию, dense, стандартный query engine;
   - `LI-tuned` — чанки 400/60, `QdrantVectorStore(enable_hybrid=True)` + fastembed BM25, bge-reranker-v2-m3 как node postprocessor, наш промпт, `MetadataFilters` по `agent_id`; бюджет на настройку ~1 день;
   - `LI-readers` — `LI-tuned` с ридерами LlamaIndex вместо наших парсеров (без FB2).
2. Общие условия: тот же корпус, `dataset_sha`, bge-m3 через тот же Ollama, `deepseek-flash` с `temperature=0` и одинаковым `LLM_REASONING_EFFORT`, тот же judge, `final_k=8`, отдельные коллекции `eval_li__*`.
3. Адаптер, чтобы baselines выдавали результат в формате `QueryResult` (цитаты `[n]` → чанки) и считались тем же кодом метрик.
4. `make compare AGENT=tolstoy` — прогон 4 участников и отчёт.
5. Качественный разбор:
   - LOC ядра против LOC обвязки LlamaIndex;
   - что пришлось переопределять во фреймворке (tenant-фильтр в hybrid, цитаты с главами, отказ до LLM, стрим в SSE);
   - 5 примеров, где выигрывает каждый участник.
6. Вывод по ADR-3 (подтверждён / пересмотрен / частично — например, ридеры LlamaIndex оказались не хуже) — правка ADR-3 и SPEC §9.

**Критерии готовности / демо**
- В git закоммичен `reports/eval/<date>_core_vs_llamaindex.md` с **таблицей метрик**. Строки — `LI-default`, `LI-tuned`, `LI-readers`, `core-best`. Столбцы — recall@8, MRR@10, faithfulness, answer relevancy, context precision, refusal F1, citation validity, TTFT p50/p95, input-токены на ответ, $ на 100 ответов.
- Для `core-best − LI-tuned` посчитаны 95 % CI разниц ключевых метрик (recall@8, faithfulness, refusal F1).
- ADR-3 обновлён выводом со ссылкой на отчёт. Строка с Δ в SPEC §9 заполнена.
- Прогон воспроизводим: повторный `make compare` на том же `dataset_sha` даёт метрики retrieval бит-в-бит, а LLM-метрики — в пределах CI.

---

## Итерация 7 — LLM-адаптер: мультипровайдер, fallback, кэш, экономия (3 дня)

**Цель.** Ответы переживают падение DeepSeek, стоимость видна и снижена.

**Задачи**
1. `AnthropicProvider` (Messages API, stream, tools, `cache_control` на системный блок), `OpenAICompatProvider` для OpenAI и Ollama (на Windows-хосте), общий `LLMChunk` / `ToolCall`.
2. `LLMRouter`: цепочка из конфига, fallback до первого токена, circuit breaker в Redis, семафоры, обработка context length.
3. Учёт токенов и стоимости (`llm_prices.yaml`), `cached_input_tokens` для DeepSeek (prefix cache) и Anthropic.
4. Кэши: эмбеддинг запроса, retrieval, ответ (ключи §11), инвалидация через `corpus_version`. Эффективный порог отказа входит в `settings_hash`.
5. Rate limit по токенам в сутки на пользователя (мягкий лимит + сообщение в UI).
6. Contract-тесты провайдеров на записанных стримах (`respx`, фикстуры SSE-ответов) без сети.

**Критерии готовности / демо**
- В `.env` ставлю неверный ключ DeepSeek → ответ приходит от Claude, в UI бейдж провайдера, в трейсе span fallback, breaker `open`.
- Повторный вопрос → ответ из кэша за < 100 мс. После загрузки нового документа кэш не используется.
- Отчёт: средние токены и $ на ответ до и после (prefix cache, бюджет контекста).

---

## Итерация 8 — Чаты и агентный режим (3–4 дня) → **MVP**

**Цель.** Диалог с историей и multi-hop вопросы через function calling.

**Задачи**
1. `chats` / `messages` в UI: список чатов агента, продолжение, удаление.
2. Condense follow-up вопроса (дешёвая модель), `standalone_question` сохраняется; `chat.summary` для длинных чатов.
3. Агентный режим (переключатель в настройках агента): tool `search_sources(query, document_ids?)`, ≤ 3 шагов, затем стрим финального ответа. Цитаты собираются из всех шагов, порог отказа применяется к каждому шагу. Все шаги видны в трейсе.
4. Фильтр по документам в UI («искать только в…»).
5. Eval: категория `multi_hop` в обычном и агентном режиме, сравнение качества, латентности и стоимости.
6. Ревизия SPEC §8: прогон всех критериев приёмки MVP, README с демо-гифкой и таблицей лицензий моделей.

**Критерии готовности / демо**
- «А почему он так считал?» после вопроса о смысле жизни → корректный ответ в контексте, standalone-вопрос виден в трейсе.
- «Сравни взгляды на смерть в „Исповеди“ и „Смерти Ивана Ильича“» → в агентном режиме 2 вызова `search_sources`, цитаты из обеих книг.
- Все пункты SPEC §8 выполнены.

---

## Итерация 9 — Hardening и эксплуатация (3–4 дня)

**Цель.** Поведение под нагрузкой и при сбоях подтверждено цифрами.

**Задачи**
1. Resumable SSE: генерация в фоне → Redis Stream → SSE с `Last-Event-ID`.
2. Нагрузочный тест (locust): 5–10 одновременных вопросов со стримом, отдельно — **с идущим ingest** (reranker и Ollama делят CPU). Фиксация TTFT p95 и узких мест.
3. Chaos-сценарии как скрипты: kill воркера, stop Qdrant или RabbitMQ посреди ingest, недоступный LLM.
4. Опционально PostgreSQL RLS для таблиц с `agent_id`.
5. Профиль `monitoring` (Prometheus + Grafana, маленький дашборд), если позволяет память. Иначе — снапшоты `/metrics`.
6. Эксперимент semantic cache: порог, доля ложных попаданий на eval.
7. `make backup` / `make restore` (pg_dump + snapshot Qdrant + uploads).

**Критерии готовности / демо**
- Отчёт нагрузочного теста с графиком латентности (с ingest и без).
- F5 во время генерации → ответ продолжается с того же места.

---

## Итерация 10 — Расширения: Streamlit + MCP (обязательно), граф / vLLM / k8s (по выбору) (3 + 5 дней)

**Цель.** Инструменты разработчика, интеграция с экосистемой агентов, направления роста.

**Задачи — обязательная часть**
1. Streamlit (`admin/`, профиль `admin`, порт 8501 на localhost): вопрос → таблица кандидатов (dense, sparse, RRF, rerank ранги и скоры), эффективный порог, итоговый контекст, промпт, ответ; сравнение двух конфигураций бок о бок; просмотр eval-прогонов и кривой калибровки.
2. MCP-сервер (`mcp` Python SDK, streamable HTTP, auth по API-ключу): `list_agents`, `search_sources`, `ask_agent`. Подключение к Claude Code или Claude Desktop, инструкция в README.

**Задачи — по выбору (1–3)**
3. **Граф знаний:** LLM-извлечение сущностей и отношений на ingest (очередь `ingest.graph`), таблицы `entities` / `relations` (или Memgraph), retrieval «сущности вопроса → связанные чанки». Eval на `multi_hop`.
4. **vLLM:** docker-профиль для GPU-хоста, провайдер `vllm` в цепочке, сравнение качества и цены с DeepSeek на eval.
5. **k8s:** kustomize-манифесты (web, reranker, воркеры, beat), KEDA ScaledObject по глубине очередей, Secrets, HPA для web, readiness и liveness. Проверка в kind или k3d (на отдельной машине: памяти VM не хватит).

**Критерии готовности / демо**
- В Streamlit видно, почему конкретный чанк не попал в top-8.
- В Claude Code: «спроси агента Толстого, что он думал о непротивлении» → ответ с цитатами через MCP.
- Для графа: пример multi-hop вопроса, который граф улучшил, с цифрами eval. Для k8s: `kubectl apply -k deploy/k8s/overlays/dev` в kind поднимает приложение, KEDA масштабирует `worker-embed` при заливке книги.

---

## Сводка

| # | Итерация | Демо-результат | Дни |
|---|---|---|---|
| 1 | Сквозной скелет | txt → вопрос → стрим DeepSeek | 3–4 |
| 2 | Auth, API, статусы, CI | логин, API-ключ, живые статусы | 3 |
| 3 | Ingest production-grade | 5 форматов, chaos-тест, DLQ | 5–6 |
| 4 | Eval + наблюдаемость | baseline `dense` в цифрах, трейсы | 3–4 |
| 5 | Гибрид, rerank, порог, цитаты | таблица улучшений, откалиброванный порог | 5 |
| 6 | Своё ядро vs LlamaIndex | таблица метрик с CI, вывод по ADR-3 | 3 |
| 7 | LLM-адаптер, fallback, кэш | переживает падение DeepSeek | 3 |
| 8 | Чаты, агентный режим | **MVP** | 3–4 |
| 9 | Hardening | нагрузочный отчёт, resumable SSE | 3–4 |
| 10 | Streamlit + MCP; граф / vLLM / k8s | RAG-debugger, агент в Claude Code | 3 + 5 |
