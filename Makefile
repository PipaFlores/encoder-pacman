all: format lint test

format:
	ruff format --exclude "*.ipynb"

lint:
	ruff check --fix-only --exclude "*.ipynb"

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

# Content-fingerprinted make_data output and validation encodings (fit(ignore_cache=True) rewrites them)
clean-cache:
ifeq ($(OS),Windows_NT)
	if exist cache rmdir /s /q cache
else
	rm -rf cache/
endif

# Reader preprocessing (pellet maps, A* distances); rebuilding it takes several minutes
clean-gamestate:
ifeq ($(OS),Windows_NT)
	if exist data\gamestate.pkl del /f /q data\gamestate.pkl
else
	rm -f data/gamestate.pkl
endif

clean-all: clean clean-cache clean-gamestate

