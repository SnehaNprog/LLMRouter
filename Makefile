.PHONY: install dev redis-up redis-down test-redis eval-baseline eval-router eval-score eval-report eval-models eval-sweep migrate migration admin-key test lint fmt typecheck check

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
