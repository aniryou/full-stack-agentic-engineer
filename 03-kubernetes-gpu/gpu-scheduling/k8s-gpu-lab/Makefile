PY ?= python3
S ?= s1

.PHONY: setup test notebooks check lab clean validate check-crds \
        kind-up kind-down kind-scenario kind-predict kind-status kind-kwok tf-validate

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

lab:
	$(PY) -m jupyterlab notebooks

# Schema checks for every manifest under deploy/: core kinds against Kubernetes 1.34 (strict),
# CRD kinds against the pinned upstream CRDs (downloads them; needs network).
validate:
	kubernetes-validate --strict --quiet -k 1.34.0 $$(find deploy -name '*.yaml' | sort)
	bash -n deploy/lib.sh deploy/kind/*.sh deploy/gke/*.sh deploy/gpu-vm/*.sh

check-crds:
	$(PY) tools/check_crds.py

# ---- the kind path (a laptop with Docker, $0) -------------------------------------------
kind-up:            ## create the cluster, advertise fake GPUs, install Kueue/JobSet/LWS, apply the queues
	deploy/kind/up.sh

kind-predict:       ## what the bundled predictor expects for scenario $(S)
	$(PY) -m k8sgpu kind predict $(S)

kind-scenario:      ## run scenario $(S) on the kind cluster and compare with the prediction
	$(PY) -m k8sgpu kind run $(S)

kind-status:
	$(PY) -m k8sgpu kind status

kind-kwok:          ## optional: 32 fake 8-GPU nodes with KWOK for placement at scale
	deploy/kind/kwok.sh

kind-down:
	deploy/kind/down.sh

tf-validate:
	cd deploy/gcp/terraform && terraform fmt -check -recursive && terraform init -backend=false -input=false >/dev/null && terraform validate

clean:
	rm -rf .pytest_cache **/__pycache__ k8sgpu/__pycache__ tests/__pycache__ **/.ipynb_checkpoints _run_outputs *.egg-info \
	       deploy/gcp/terraform/.terraform deploy/gcp/terraform/.terraform.lock.hcl
