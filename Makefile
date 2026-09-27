SHELL := /bin/bash
UV ?= $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
COMPOSE := docker compose
TEST_COMPOSE := docker compose -f compose.test.yaml
TEST_ENV := DATABASE_URL=postgresql+asyncpg://rag_test:rag_test@127.0.0.1:15432/rag_test \
            QDRANT_URL=http://127.0.0.1:16333 APP_ENV=test

.PHONY: help up up-debug down build logs ps migrate seed sh lint fmt test test-unit test-integration test-up test-down smoke

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

logs: ## логи сервиса: make logs s=worker
	$(COMPOSE) logs -f --tail=200 $(s)

ps: ## состояние сервисов и память
	$(COMPOSE) ps
	@docker stats --no-stream --format 'table {{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}' | grep rag-agents || true

migrate: ## alembic upgrade head
	$(COMPOSE) run --rm migrate

seed: ## seed-пользователь (владелец агентов до итерации 2)
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
