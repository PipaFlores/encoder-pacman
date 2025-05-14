all: format lint test

format:
	ruff format

lint:
	ruff check --fix-only

test:
	pytest -v

# install:
# 	python -m pip install -e .


