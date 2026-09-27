PY ?= python3

.PHONY: setup test notebooks check lab stack clean

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
	bash -n deploy/local/*.sh deploy/any-gpu/*.sh

lab:
	$(PY) -m jupyterlab notebooks

# T0 playground: the gateway and two fake providers in one process, on fixed ports (Ctrl-C to stop).
stack:
	$(PY) -m gwlab stack --port 8080

clean:
	rm -rf .pytest_cache _run_outputs *.egg-info build dist
	find . \( -name __pycache__ -o -name .ipynb_checkpoints -o -name '.terraform*' -o -name '.gwlab-*' \) -prune -exec rm -rf {} +
