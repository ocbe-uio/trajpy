.PHONY: check-format fix test lint type-check security all

check-format:
	.venv/bin/ruff format trajpy --check

fix:
	.venv/bin/ruff format trajpy
	.venv/bin/ruff check trajpy --fix

test:
	.venv/bin/python -m pytest -s tests

lint:
	.venv/bin/ruff check trajpy

type-check:
	.venv/bin/pyright trajpy

security:
	.venv/bin/bandit -r trajpy

all: lint type-check check-format fix test security
