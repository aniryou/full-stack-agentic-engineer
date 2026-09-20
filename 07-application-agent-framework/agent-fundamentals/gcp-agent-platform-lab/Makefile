PY ?= python3

.PHONY: setup test test-all notebooks check lab clean

## Install the package (editable) and the dev tools
setup:
	$(PY) -m pip install -e ".[dev]"
	$(PY) -m ipykernel install --user --name python3 --display-name "Python 3 (agentlab)"

## Unit tests for the library (fast)
test:
	$(PY) -m pytest -q -m "not slow"

## Regenerate notebooks/ (exercises) and solutions/ from notebooks_src/
notebooks:
	$(PY) tools/build_notebooks.py

## Everything: rebuild notebooks, run unit tests, execute every solution notebook,
## and confirm the exercise variants stop at the first unsolved cell
check: notebooks
	$(PY) -m pytest -q -m "not slow"
	$(PY) tools/run_notebooks.py solutions
	$(PY) tools/run_notebooks.py notebooks --expect-fail

## Unit tests plus the slow notebook-execution tests through pytest
test-all:
	$(PY) -m pytest -q

## Open JupyterLab in the exercises folder
lab:
	$(PY) -m jupyterlab notebooks

clean:
	rm -rf .pytest_cache build dist *.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name ".ipynb_checkpoints" -type d -prune -exec rm -rf {} +
