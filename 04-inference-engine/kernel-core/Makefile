PY ?= python3
KV := ../kv-cache

.PHONY: setup test notebooks check lab clean

setup:
	$(PY) -m pip install -e '.[dev]'
	$(PY) -m ipykernel install --user --name python3

test:
	$(PY) -m pytest -q

# Re-execute the worked notebook and save its outputs (the kv-cache notebooks are hand-written .ipynb).
notebooks:
	$(PY) tools/run_notebooks.py $(KV)/01_kv_cache_worked.ipynb --write

# Tests, then: the worked notebook runs clean and the practice blank stops at its first blank.
check:
	$(PY) -m pytest -q
	$(PY) tools/run_notebooks.py $(KV)/01_kv_cache_worked.ipynb
	$(PY) tools/run_notebooks.py $(KV)/02_kv_cache_practice.ipynb --expect-fail

lab:
	$(PY) -m jupyterlab $(KV)

clean:
	rm -rf .pytest_cache **/__pycache__ **/.ipynb_checkpoints _run_outputs *.egg-info
