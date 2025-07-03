all: format lint test

format:
	ruff format --exclude notebook,playground

lint:
	ruff check --fix-only --exclude notebook,playground

test:
	pytest -v


clean:
ifeq ($(OS),Windows_NT)
	if exist temp\* del /f /q temp\*
	if exist src\temp\* del /f /q src\temp\*
else
	find temp/ -type f -delete
	find src/temp/ -type f -delete
endif

