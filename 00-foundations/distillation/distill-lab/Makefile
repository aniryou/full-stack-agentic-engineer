PY ?= python3

.PHONY: setup test notebooks check lab fake tinylm clean

setup:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .
	$(PY) -m ipykernel install --user --name python3

test:
	$(PY) -m pytest -q

notebooks:
	$(PY) tools/build_notebooks.py

check: notebooks
	$(PY) -m pytest -q
	$(PY) tools/run_notebooks.py solutions
	$(PY) tools/run_notebooks.py notebooks --expect-fail
	bash -n deploy/any-gpu/*.sh

lab:
	$(PY) -m jupyterlab notebooks

# T0 playground: a fake vLLM teacher on :8000 (every answer simulated).
fake:
	$(PY) -m distillab fake --port 8000

tinylm:
	$(PY) -m distillab tinylm

clean:
	rm -rf .pytest_cache _run_outputs *.egg-info build dist
	find . \( -name __pycache__ -o -name .ipynb_checkpoints -o -name '.terraform*' \) -prune -exec rm -rf {} +
