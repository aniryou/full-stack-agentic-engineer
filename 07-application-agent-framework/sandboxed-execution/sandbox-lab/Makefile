PY ?= python3

.PHONY: setup test notebooks check lab probes render clean

setup:
	$(PY) -m pip install -e ".[dev]"
	$(PY) -m ipykernel install --user --name python3

test:
	$(PY) -m pytest -q

notebooks:
	$(PY) tools/build_notebooks.py

render:
	$(PY) tools/render_manifests.py

check: render notebooks
	$(PY) tools/render_manifests.py --check
	$(PY) -m pytest -q
	$(PY) tools/run_notebooks.py solutions
	$(PY) tools/run_notebooks.py notebooks --expect-fail
	bash -n deploy/lib.sh deploy/docker/*.sh deploy/kind/*.sh deploy/gke/*.sh deploy/gcp/*.sh

# T0 playground: the attack probes through a process sandbox, then the whole ladder.
probes:
	$(PY) -m sandboxlab env
	$(PY) -m sandboxlab probes --level process+netns

lab:
	$(PY) -m jupyterlab notebooks

clean:
	rm -rf .pytest_cache _run_outputs reports *.egg-info build dist
	find . \( -name __pycache__ -o -name .ipynb_checkpoints -o -name '.terraform*' -o -name '*.secret' \) -prune -exec rm -rf {} +
