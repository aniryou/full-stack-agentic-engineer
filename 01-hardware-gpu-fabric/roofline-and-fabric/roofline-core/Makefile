PY ?= python3

.PHONY: setup setup-notebooks test notebooks check lab clean

setup:            ## enough for the tests
	$(PY) -m pip install -r requirements.txt

setup-notebooks:  ## JupyterLab and the notebook runner (`make check`, `make lab`)
	$(PY) -m pip install -r requirements-notebooks.txt
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
