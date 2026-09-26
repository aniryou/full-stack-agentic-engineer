PY ?= python3

.PHONY: setup test notebooks check lab fixtures record clean

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
	MOELAB_NO_TORCH=1 $(PY) tools/run_notebooks.py solutions
	bash -n deploy/any-gpu/*.sh deploy/gke/*.sh
	kubernetes-validate --strict -k 1.34.0 deploy/gke/*.yaml

lab:
	$(PY) -m jupyterlab notebooks

# Regenerate the illustrative fixtures (seconds, no torch) / the recorded tiny-MoE curves (torch, ~1 min).
fixtures:
	$(PY) tools/make_fixtures.py

record:
	$(PY) -m moelab.tinymoe --record

clean:
	rm -rf .pytest_cache _run_outputs *.egg-info build dist results deploy/any-gpu/results deploy/gke/out
	find . \( -name __pycache__ -o -name .ipynb_checkpoints \) -prune -exec rm -rf {} +
