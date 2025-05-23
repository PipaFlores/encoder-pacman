all: format lint test clean_temp_files

format:
	ruff format

lint:
	ruff check --fix-only

test:
	pytest -v


clean_temp_files:
ifeq ($(OS),Windows_NT)
	if exist temp\* del /f /q temp\*
	if exist src\temp\* del /f /q src\temp\*
else
	find temp/ -type f -delete
	find src/temp/ -type f -delete
endif

