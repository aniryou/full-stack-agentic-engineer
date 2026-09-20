.PHONY: install test lint format notebooks tf-validate demo zip

install:
	pip install -e ".[dev]"

test:
	python -m pytest -q

lint:
	ruff check src tests && ruff format --check src tests

format:
	ruff check --fix src tests && ruff format src tests

notebooks:  ## execute all worked + solution notebooks (offline)
	mkdir -p build/nb && for nb in notebooks/*.ipynb notebooks/solutions/*.ipynb; do \
	  jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=180 --output-dir build/nb "$$nb" || exit 1; done

tf-validate:
	cd infra/terraform && terraform init -backend=false -input=false >/dev/null && terraform fmt -check -recursive && terraform validate

demo:
	agentsec demo
