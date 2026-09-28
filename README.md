# RAG-агенты


Пользователь создаёт «агента» (например, «Внутренний мир Льва Толстого»), загружает в него книги и получает ответы **только по этим материалам** — со ссылками на источники и стримингом.

Стек: Python 3.12 · FastAPI + Jinja2 + HTMX (SSE) · Celery + RabbitMQ · PostgreSQL · Redis · Qdrant · Ollama `bge-m3` · DeepSeek.

Документы: [SPEC](docs/SPEC.md) · [ARCHITECTURE](docs/ARCHITECTURE.md) · [PLAN](docs/PLAN.md) · [CLAUDE.md](CLAUDE.md) (правила разработки).

**Статус:** итерация 3 — ingest production-grade: 5 форматов (txt, fb2/fb2.zip, epub, pdf, docx), структурный чанкинг по главам, очереди с фан-аутом батчей, DLQ и replay, асинхронное удаление. Итерация 2 — вход, JSON API с ключами, rate limit. Итерация 1 — сквозной скелет.

Обзор проекта со схемами (стек, workflow индексации и ответа, метрики): откройте [docs/interview/index.html](docs/interview/index.html) в браузере.

## Быстрый старт

```bash
cp .env.example .env        # заполнить DEEPSEEK_API_KEY, пароли POSTGRES_* / RABBITMQ_*, SEED_USER_PASSWORD
make up                     # сборка образа + весь стек; первый раз скачивает bge-m3 (~1.2 ГБ)
make seed                   # администратор SEED_USER_EMAIL с паролем SEED_USER_PASSWORD
```

Другие пользователи: `docker compose exec web rag-agents user create me@example.com [--admin]` (пароль спросит интерактивно), смена пароля — `rag-agents user set-password EMAIL` (все сессии пользователя завершаются).

UI: <http://192.168.56.10:8080> (с Windows-хоста) или <http://localhost:8080> (на VM).
Миграции применяются автоматически одноразовым сервисом `migrate` при `make up`; вручную — `make migrate`.

Разработка: `make lint` (ruff, mypy --strict, import-linter), `make test` (unit + integration на `compose.test.yaml`), `make smoke` (e2e по живому стенду). Нужен [uv](https://docs.astral.sh/uv/).

## Вход и JSON API

- UI требует входа (`/login`). Сессия — в Redis (7 дней, продлевается при активности), cookie `HttpOnly; SameSite=Lax`. Все изменяющие запросы UI несут CSRF-токен.
- «🔬 Под капотом» (`/system`) видят только администраторы: там живые события всех пользователей.
- API-ключи: «🔑 API-ключи» в шапке → «Выпустить ключ». Ключ `rag_<prefix>_<secret>` показывается один раз, в БД — только `sha256(secret)`. Отзыв действует сразу.
- Документация API (OpenAPI, кнопка Authorize): <http://192.168.56.10:8080/api/docs>. Ошибки — `application/problem+json` (RFC 9457).
- Лимиты: вопросы — 20 в минуту на пользователя, загрузки — 30 файлов в час, вход — 5 попыток в минуту с IP. Превышение — `429` с `Retry-After`.

```bash
KEY=rag_…   # из /settings/api-keys
curl -s -H "Authorization: Bearer $KEY" localhost:8080/api/v1/agents | jq
curl -s -H "Authorization: Bearer $KEY" -F files=@book.txt localhost:8080/api/v1/agents/$AGENT/documents
curl -N -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  localhost:8080/api/v1/agents/$AGENT/query -d '{"question": "В чём смысл жизни?", "stream": true}'
```

Без `"stream": true` ответ приходит одним JSON (`chat_id`, `message_id`, `result`). С `chat_id` вопрос продолжает существующий чат.

## Настройки LLM

В `.env`:

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `LLM_MODEL` | `deepseek-flash` | модель ответов (`deepseek-flash`, `deepseek-v4-pro`) |
| `LLM_REASONING_EFFORT` | `low` | глубина рассуждений `low` / `high` / `max`; пусто — параметр не передаётся |

Под каждым ответом в UI видно модель, effort, время первого токена и число токенов (в т. ч. reasoning) — удобно сравнивать `low` и `high`. После смены значения: `docker compose up -d web`.

## Трейсы (Langfuse Cloud)

В `.env` задайте `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` и `LANGFUSE_BASE_URL` (EU: `https://cloud.langfuse.com`; старое имя `LANGFUSE_HOST` тоже читается). Без ключей трейсинг выключен, и всё работает как раньше.

```bash
make langfuse-check              # связь: тестовый трейс отправлен и виден через API
make langfuse-model              # один раз: цена LLM_MODEL в Langfuse (для расчёта cost)
make langfuse-trace id=<trace>   # дерево span-ов, токены, cost и 👍/👎 трейса
```

Сводка — на странице **📊 Аналитика** (`/insights`): стоимость, латентность, первый токен, доля 👍, агенты, где тратится время, последние запросы с деревом шагов. Сводка считается по нашей БД (работает и без Langfuse), дерево шагов — из Langfuse по клику. Стоимость — по `configs/llm_prices.yaml` с тарифами peak/off-peak DeepSeek. Под каждым ответом — ссылка «🔎 трейс». Демо-данные: `make demo-traffic ROUNDS=2` (≈ $0.06).

Трейс создаётся на каждый вопрос (поиск, эмбеддинг, генерация DeepSeek) и на каждую индексацию документа. Кнопки 👍/👎 под ответом становятся score `user_feedback` на трейсе ответа. Подробности — [ARCHITECTURE §14.2](docs/ARCHITECTURE.md#142-трейсы--langfuse-cloud-eu).

## Как проверить итерацию 3 руками

Критерии — [PLAN, итерация 3](docs/PLAN.md#итерация-3--ingest-production-grade-форматы-структура-очереди-56-дней).

1. **Сервисы:** `docker compose ps` — `worker-ingest`, `worker-embed`, `beat` вместо одного `worker`. RabbitMQ UI → Queues: `ingest.parse`, `ingest.embed`, `maintenance`, `eval` и их `*.dlq`.
2. **Форматы.** В агента загрузить книги разных форматов (примеры — `tests/fixtures/`: fb2, fb2.zip, epub, pdf, docx, txt в cp1251). В строке документа видно `processing · embedding · батчей N/M`, затем `done`.
3. **Оглавление и чанки:** клик по названию документа → страница с метаданными (источник оглавления, кодировка, страницы, тайминги), оглавлением по главам и чанками постранично (номера страниц PDF, число токенов).
4. **Ошибки без ретраев:** `tests/fixtures/broken.pdf` → `failed: Файл повреждён (corrupted)`, `scan.pdf` → `no_text_layer`. `.docx`, переименованный в `.pdf`, отклоняется при загрузке.
5. **DLQ и replay:** `make chaos-embed`, загрузить документ → он `failed (internal_error)`, в RabbitMQ `ingest.embed.dlq` — 1 сообщение. `make dlq-replay QUEUE=ingest.embed` → документ становится `done`.
6. **Chaos-kill:** загрузить большую книгу, посреди эмбеддинга `docker kill rag-agents-worker-embed-1`, затем `docker compose up -d worker-embed` — документ доезжает до `done`, число точек в Qdrant равно `chunks_total` (сообщения вернулись в очередь благодаря `acks_late`, дубли отсекает учёт батчей).
7. **Удаление:** ✕ у документа — строка исчезает сразу, через секунды точки удалены из Qdrant (карта векторов на `/system`), файл — с диска. Удаление агента чистит всё в фоне.
8. **Sweeper:** `make logs s=beat` — раз в минуту `maintenance.sweep`; застрявшие документы переотправляются сами.

## Как проверить итерацию 2 руками

Критерии готовности — [PLAN, итерация 2](docs/PLAN.md#итерация-2--фундамент-auth-api-статусы-ci-3-дня).

1. **Вход и выход.** Открыть <http://192.168.56.10:8080> → редирект на `/login`. Войти как `SEED_USER_EMAIL` / `SEED_USER_PASSWORD` → список агентов (агенты итерации 1 принадлежат этому пользователю). «Выйти» → снова `/login`; кнопка «Назад» в браузере данных не показывает.
2. **Чужой агент → 404.** `rag-agents user create other@local`, войти им в другом браузере (или инкогнито) и открыть ссылку на агента первого пользователя — «404 — Не найдено». Все эндпоинты с чужими ID обходит `tests/integration/test_isolation.py`.
3. **API-ключ и стрим JSON.** Выпустить ключ на `/settings/api-keys` и выполнить `curl -N … /query` из раздела выше — события `sources`, `token`…, `done` с JSON в `data:`. Отозвать ключ → тот же запрос получает `401`.
4. **Живые статусы.** Загрузить 3 файла → строки сами проходят `queued` → `processing` (этап, прогресс) → `done`. DevTools → Network: запросы `/documents/status` раз в 2 с, последний — `286`, дальше тишина. У `failed` есть кнопка ↻ (повтор), у завершённых — ✕ (удаление из PG, Qdrant и диска).
5. **Rate limit.** 6 неверных паролей подряд → «Слишком много запросов. Повторите через N с.».
6. **Логи с request_id.** `make logs s=web` — JSON-строки `http.request` с `request_id`, `user_id`, `status`, `duration_ms`; заголовок `X-Request-ID` в ответе.
7. **Автопроверка:** `make lint test smoke` — smoke теперь логинится, а в конце задаёт тот же вопрос через JSON API по ключу.

## Как проверить итерацию 1 руками

Критерии готовности — [PLAN, итерация 1](docs/PLAN.md#итерация-1--сквозной-скелет-34-дня).

1. **Стенд поднимается.**
   ```bash
   make up && make seed
   docker compose ps          # postgres, redis, rabbitmq, qdrant, ollama, web, worker — healthy/running;
                              # migrate и ollama-pull — exited (0)
   curl -s localhost:8080/readyz   # {"ready": true, "checks": {... все "ok"}}
   ```
2. **UI открывается с Windows**: <http://192.168.56.10:8080> → страница «Агенты».
3. **Создать агента**: «Создать агента» → название «Толстой (тест)», описание по желанию → «Создать».
4. **Загрузить txt.** Любой русский `.txt` (UTF-8 или cp1251). Имя вида `Автор - Название.txt` даст красивую подпись источника (`Л. Н. Толстой — «Исповедь»`).
   Пример — «Исповедь» с az.lib.ru, конвертация в txt в cp1251 (заодно проверка определения кодировки):
   ```bash
   mkdir -p data/samples && curl -s http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0440.shtml \
     | python3 scripts/lib_ru_to_txt.py > "data/samples/Л. Н. Толстой - Исповедь.txt"
   ```
   В таблице документов статус меняется сам (polling раз в 2 с): `queued` → `processing` (этап и прогресс-бар) → `done` с числом чанков и временем обработки. Ожидаемо — ≤ 5 мин на CPU. Когда все файлы в конечном статусе, polling останавливается (DevTools → Network: последний ответ `/documents/status` со статусом 286).
5. **Задать вопрос** «В чём смысл жизни?»: сначала появляется «ищу в источниках…», затем блок «Источники (6)», затем ответ печатается потоком; после окончания текст форматируется, ссылки `[n]` ведут к карточкам источников, внизу — модель, effort, время первого токена, токены.
6. **Вопрос вне корпуса** («Какой курс биткоина?») — агент должен ответить «В материалах агента я не нашёл ответа…» (в итерации 1 отказ формулирует LLM по правилу промпта; отказ до LLM по порогу — итерация 5).
7. **Перезагрузка страницы** — история вопросов и ответов на месте (хранится в PostgreSQL).
8. **Повторная загрузка того же файла** — сообщение «уже загружен… пропущен», дубликата нет.
9. **Изоляция**: создать второго агента, загрузить другой текст — его ответы и источники только из своего файла. Автоматически это проверяет `tests/integration/test_isolation.py`.
10. **Очереди**: RabbitMQ UI на порту 15672 (логин/пароль из `.env`) → Queues: `ingest.parse`, `ingest.embed`, `maintenance`, `eval` и их `*.dlq`.
    **Под капотом**: <http://192.168.56.10:8080/system> — живая схема сервисов (анимируется при вопросе и загрузке), карта векторов Qdrant, очереди, воркеры, Redis, PostgreSQL, Ollama и справочник технологий. `make up-debug` поднимает веб-админки (Flower, pgweb, RedisInsight), ссылки на них и на Qdrant Dashboard и RabbitMQ UI есть на этой странице. Чтобы они открывались с Windows, в `.env` нужно `ADMIN_UI_BIND=192.168.56.10`.
11. **Память**: `make ps` — потребление в пределах лимитов (ARCHITECTURE §13).
12. **Автопроверка всего сразу**: `make lint test smoke`.

Упрощения итерации 1 (по PLAN; авторизация появилась в итерации 2): только `.txt`, чанкинг по абзацам без учёта глав, dense-поиск top-6 без reranker, один воркер на все очереди, один чат на агента.
