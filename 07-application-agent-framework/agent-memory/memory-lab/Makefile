PY ?= python3

.PHONY: setup test notebooks check lab demo clean

setup:
	$(PY) -m pip install -e ".[dev]"
	$(PY) -m ipykernel install --user --name python3

test:
	$(PY) -m pytest -q

notebooks:
	$(PY) tools/build_notebooks.py

check: notebooks
	$(PY) -m pytest -q
	$(PY) tools/run_notebooks.py solutions
	$(PY) tools/run_notebooks.py notebooks --expect-fail
	bash -n deploy/local/up.sh deploy/local/down.sh

# T0 playground: write, search and forget in a SQLite store, then look for the bytes on disk.
demo:
	$(PY) -m memlab env
	$(PY) -m memlab demo

lab:
	$(PY) -m jupyterlab notebooks

clean:
	rm -rf .pytest_cache _run_outputs reports *.egg-info build dist
	find . \( -name __pycache__ -o -name .ipynb_checkpoints -o -name '*.db' -o -name '*.db-wal' -o -name '*.db-shm' \) -prune -exec rm -rf {} +
