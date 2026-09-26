PY ?= python3

.PHONY: setup test notebooks check lab fake bench size clean

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
	bash -n deploy/any-gpu/*.sh deploy/gcp/cloud-run/*.sh deploy/gcp/gke/*.sh

lab:
	$(PY) -m jupyterlab notebooks

# T0 playground: a fake vLLM on :8000, then a short benchmark against it.
fake:
	$(PY) -m servelab fake --profile t4-qwen2.5-0.5b --port 8000

bench:
	$(PY) -m servelab bench --url http://127.0.0.1:8000 --num-requests 50 --rate 5

size:
	$(PY) -m servelab size --model qwen2.5-0.5b-instruct --gpu T4 --max-model-len 4096

clean:
	rm -rf .pytest_cache **/__pycache__ */*/__pycache__ **/.ipynb_checkpoints _run_outputs *.egg-info
