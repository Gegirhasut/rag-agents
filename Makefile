SHELL := /bin/bash
UV ?= $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
COMPOSE := docker compose
TEST_COMPOSE := docker compose -f compose.test.yaml
TEST_ENV := DATABASE_URL=postgresql+asyncpg://rag_test:rag_test@127.0.0.1:15432/rag_test \
            QDRANT_URL=http://127.0.0.1:16333 REDIS_URL=redis://127.0.0.1:16379/0 APP_ENV=test

.PHONY: eval eval-diff eval-list eval-export metrics dlq-replay chaos-embed help up up-debug down build logs ps migrate seed sh lint fmt test test-unit test-integration test-up test-down smoke demo-traffic langfuse-check langfuse-model langfuse-trace

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-18s %s\n", $$1, $$2}'

up: ## поднять стенд (собирает образ при необходимости)
	$(COMPOSE) up -d --build

up-debug: ## стенд + веб-админки (Flower, pgweb, RedisInsight); ссылки на /system
	$(COMPOSE) --profile debug up -d --build

down: ## остановить стенд (данные в volumes сохраняются)
	$(COMPOSE) --profile debug down

build: ## пересобрать образ приложения
	$(COMPOSE) build web

logs: ## логи сервиса: make logs s=worker-ingest
	$(COMPOSE) logs -f --tail=200 $(s)

ps: ## состояние сервисов и память
	$(COMPOSE) ps
	@docker stats --no-stream --format 'table {{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}' | grep rag-agents || true

migrate: ## alembic upgrade head
	$(COMPOSE) run --rm migrate

seed: ## seed-администратор SEED_USER_EMAIL с паролем SEED_USER_PASSWORD из .env
	$(COMPOSE) exec web rag-agents seed

sh: ## shell в контейнере web
	$(COMPOSE) exec web bash

lint: ## ruff + mypy --strict + import-linter
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run mypy
	$(UV) run lint-imports

fmt: ## автоформатирование
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

test-unit: ## unit-тесты без docker
	$(UV) run pytest tests/unit -q

test-up:
	$(TEST_COMPOSE) up -d --wait

test-down:
	$(TEST_COMPOSE) down -v

test-integration: test-up ## integration-тесты на compose.test.yaml
	$(TEST_ENV) $(UV) run alembic upgrade head
	$(TEST_ENV) $(UV) run pytest tests/integration -q

test: test-unit test-integration ## все тесты

smoke: ## e2e по живому стенду: агент → txt → done → вопрос → стрим
	$(UV) run python scripts/smoke.py $(if $(FILE),--file "$(FILE)",)

demo-traffic: ## демо-агенты + ~16 вопросов × ROUNDS с 👍/👎 (для Langfuse и /insights, ≈ $0.03 за раунд)
	$(UV) run python scripts/demo_traffic.py --rounds $(or $(ROUNDS),1)

langfuse-check: ## Langfuse: auth + тестовый трейс, ждём его появления в API
	$(UV) run python scripts/langfuse_check.py ping

langfuse-model: ## Langfuse: завести цену LLM_MODEL (идемпотентно)
	$(UV) run python scripts/langfuse_check.py ensure-model

langfuse-trace: ## Langfuse: дерево, токены и cost трейса: make langfuse-trace id=<trace_id>
	$(UV) run python scripts/langfuse_check.py trace $(id)

dlq-replay: ## вернуть сообщения из DLQ в очередь: make dlq-replay QUEUE=ingest.embed
	$(COMPOSE) exec worker-ingest rag-agents dlq replay $(QUEUE)

chaos-embed: ## следующий батч эмбеддинга упадёт → ingest.embed.dlq (демо replay)
	$(COMPOSE) exec worker-ingest rag-agents chaos-embed

# --- Eval (ARCHITECTURE §15): одноразовый контейнер eval (профиль tools), отчёты в reports/eval
EVAL_RUN := $(COMPOSE) --profile tools run --rm --user $$(id -u):$$(id -g) -e GIT_SHA=$$(git rev-parse --short HEAD) eval

eval: ## eval-прогон: make eval AGENT=tolstoi DATASET=tolstoy CONFIG=dense [LIMIT=5] [JUDGE=0]
	@mkdir -p reports/eval
	$(EVAL_RUN) run --agent $(or $(AGENT),tolstoi) --dataset eval/datasets/$(or $(DATASET),tolstoy).jsonl \
		--config configs/eval/$(or $(CONFIG),dense).yaml --limit $(or $(LIMIT),0) $(if $(filter 0,$(JUDGE)),--no-judge,)

eval-diff: ## сравнить два прогона (95 % CI): make eval-diff A=<run_id> B=<run_id>
	@mkdir -p reports/eval
	$(EVAL_RUN) diff $(A) $(B)

eval-export: ## дозалить прогон в Langfuse Datasets: make eval-export RUN=<run_id>
	$(EVAL_RUN) export-langfuse $(RUN)

eval-list: ## последние прогоны агента: make eval-list AGENT=tolstoi
	$(EVAL_RUN) list --agent $(or $(AGENT),tolstoi)

metrics: ## метрики Prometheus стенда (curl /metrics, только наши ряды)
	@curl -s localhost:8080/metrics | grep -E '^(http_requests_total|rag_|llm_|ingest_|embed_batch|celery_queue_depth)' | grep -v '_created' | head -60
