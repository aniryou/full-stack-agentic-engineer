PY ?= python3

.PHONY: setup test notebooks check lab plan kv fake bench tiny clean

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
	bash -n deploy/any-gpu/*.sh deploy/gcp/*.sh

lab:
	$(PY) -m jupyterlab notebooks

# T0 playground
plan:
	$(PY) -m quantlab plan --gpu L4

kv:
	$(PY) -m quantlab kv --model llama-3.1-8b-instruct --gpu L4

fake:
	$(PY) -m quantlab fake --model qwen2.5-1.5b-instruct --gpu L4 --scheme w4a16 --port 8000

bench:
	$(PY) -m quantlab bench --schemes bf16,fp8,w4a16

# Re-create the bundled tiny checkpoint (needs torch; ~25 min on a busy CPU)
tiny:
	$(PY) tools/train_tiny.py

clean:
	rm -rf .pytest_cache _run_outputs out *.egg-info build dist
	find . \( -name __pycache__ -o -name .ipynb_checkpoints -o -name '.terraform*' \) -prune -exec rm -rf {} +
