.PHONY: install dev migrate migration admin-key test lint fmt typecheck check

install:
	pip install -e ".[dev]"

dev:
	uvicorn app.main:app --reload

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
