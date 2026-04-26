.PHONY: test lint ci-deps

ci-deps:
	pip install -r requirements-ci.txt

lint:
	ruff check ml_pipeline tests dags

test:
	pytest -q
