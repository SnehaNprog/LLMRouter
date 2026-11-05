.PHONY: install dev redis-up redis-down test-redis test-postgres test-slow docker-build up down logs load-test eval-baseline eval-router eval-score eval-report eval-models eval-sweep migrate migration admin-key test lint fmt typecheck check

install:
	pip install -e ".[dev]"

dev:
	uvicorn app.main:app --reload

# Redis for quotas (needs Docker)
redis-up:
	docker compose up -d redis

redis-down:
	docker compose down

# Postgres, Redis, the migrations and the gateway, as containers (needs config.yaml and .env)
docker-build:
	docker compose build

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f app

# measure the gateway against a stand-in model, no real model needed (see docs/load-test.md)
load-test:
	python -m app.loadtest run --generators 4 --json loadtest.json

# every HTTP request and Redis command is made slower, to expose tests that quietly assume speed;
# they fail here every time instead of now and then on a busy CI runner
test-slow:
	LLMROUTER_TEST_SLOW_MS=40 pytest -q

# also runs the whole suite against a real Postgres: every app gets a database of its own on the
# server, created and dropped by the tests. Start one first, e.g.
#   docker run -d --rm --name pg-test -e POSTGRES_PASSWORD=test -p 15432:5432 postgres:17-alpine
test-postgres:
	LLMROUTER_TEST_POSTGRES_URL=postgresql://postgres:test@localhost:15432/postgres pytest -q

# also runs the Redis tests against a real server (flushes database 15)
test-redis:
	LLMROUTER_TEST_REDIS_URL=redis://localhost:6379/15 pytest -q

# the always-frontier cost: needs the gateway running (make dev) and LLMROUTER_EVAL_KEY set
eval-baseline:
	python -m app.evaluation baseline

# the router's cost on the same prompts (model=auto); use a key from an unrestricted organisation
eval-router:
	python -m app.evaluation router

# grade both runs; set JUDGE to a gateway model other than the ones being compared
eval-score:
	python -m app.evaluation score eval_results/baseline.jsonl eval_results/router.jsonl --judge-model $(JUDGE) --negative-control

# every tiered model answers the benchmark once (what the sweep replays); then score them with
#   python -m app.evaluation score eval_results/model-*.jsonl --judge-model $(JUDGE)
eval-models:
	python -m app.evaluation models

# cost against quality across routing thresholds, chosen on half the prompts, judged on the rest
eval-sweep:
	python -m app.evaluation sweep --json eval_results/sweep.json

# cost saved, quality retained and latency, from the scored runs
eval-report:
	python -m app.evaluation evaluate --json eval_results/report.json

migrate:
	alembic upgrade head

# prints a new admin API key once; run `make migrate` first
admin-key:
	python -m app.cli create-admin

# usage: make migration m="add something"
migration:
	alembic revision --autogenerate -m "$(m)"

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

fmt:
	ruff check --fix .
	ruff format .

typecheck:
	mypy

check: lint typecheck test
