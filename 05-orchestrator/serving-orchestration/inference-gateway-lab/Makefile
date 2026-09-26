PY ?= python3

.PHONY: setup test notebooks check lab validate-k8s tf-validate clean

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
	for f in deploy/*/*.sh; do bash -n $$f || exit 1; done

validate-k8s:
	kubernetes-validate --strict -k 1.34.0 deploy/gke/vllm.yaml deploy/gke/hpa.yaml deploy/kind/sim-deployment.yaml

tf-validate:
	cd deploy/gcp/terraform && terraform fmt -check -recursive && terraform init -backend=false -input=false >/dev/null && terraform validate

lab:
	$(PY) -m jupyterlab notebooks

clean:
	rm -rf .pytest_cache _run_outputs *.egg-info build deploy/gcp/terraform/.terraform deploy/gcp/terraform/.terraform.lock.hcl
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name .ipynb_checkpoints -type d -prune -exec rm -rf {} +
