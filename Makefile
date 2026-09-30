.PHONY: install dev redis-up redis-down test-redis eval-baseline migrate migration admin-key test lint fmt typecheck check

install:
	pip install -e ".[dev]"

dev:
	uvicorn app.main:app --reload

# Redis for quotas (needs Docker)
redis-up:
	docker compose up -d redis

redis-down:
	docker compose down

# also runs the Redis tests against a real server (flushes database 15)
test-redis:
	LLMROUTER_TEST_REDIS_URL=redis://localhost:6379/15 pytest -q

# the always-frontier cost: needs the gateway running (make dev) and LLMROUTER_EVAL_KEY set
eval-baseline:
	python -m app.evaluation baseline

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
