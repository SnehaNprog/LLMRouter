.PHONY: install dev test lint fmt typecheck check

install:
	pip install -e ".[dev]"

dev:
	uvicorn app.main:app --reload

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
