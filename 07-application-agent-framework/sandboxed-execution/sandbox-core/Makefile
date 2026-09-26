PY ?= python3

.PHONY: setup test notebooks check lab clean

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

lab:
	$(PY) -m jupyterlab notebooks

clean:
	rm -rf .pytest_cache **/__pycache__ **/.ipynb_checkpoints _run_outputs *.egg-info
