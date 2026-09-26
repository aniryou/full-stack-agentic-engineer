PY ?= python3

.PHONY: setup test notebooks check lab gpu-bench dist-bench fixtures ptx-check k8s-validate tf-validate clean

setup:
	$(PY) -m pip install -r requirements.txt
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

gpu-bench:  ## T1: time every kernel on the GPU you are on (needs numba-cuda and a driver)
	$(PY) -m gpurt.kernels.bench

dist-bench:  ## T0 by default (a ring over OS pipes); BACKEND=gloo|nccl NRANKS=2 with torch
	$(PY) -m gpurt.dist.bench --backend $(or $(BACKEND),pipes) --nranks $(or $(NRANKS),2) -e 16M

fixtures:  ## regenerate the illustrative nccl-tests samples in gpurt/fixtures/
	$(PY) tools/make_fixtures.py

ptx-check:  ## compile every kernel to PTX for sm_75/sm_89 (needs numba-cuda, not a GPU); exit 2 = skipped
	$(PY) tools/check_ptx.py

k8s-validate:
	kubernetes-validate --strict -k 1.34.0 deploy/gke/0*.yaml

tf-validate:
	cd deploy/gcp/terraform && terraform init -backend=false -input=false >/dev/null && terraform fmt -check -recursive && terraform validate

clean:
	rm -rf .pytest_cache _run_outputs *.egg-info deploy/gcp/terraform/.terraform deploy/gcp/terraform/.terraform.lock.hcl
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name .ipynb_checkpoints -type d -prune -exec rm -rf {} +
