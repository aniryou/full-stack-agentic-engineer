.PHONY: install test lint format notebooks tf-validate demo zip

install:
	pip install -e ".[dev]"

test:
	python -m pytest -q

lint:
	ruff check src tests && ruff format --check src tests

format:
	ruff check --fix src tests && ruff format src tests

notebooks:  ## execute the worked notebooks and the solutions (offline); the *_practice blanks are left out
	mkdir -p build/nb/notebooks build/nb/solutions && for nb in $$(ls notebooks/*.ipynb | grep -v '_practice\.ipynb$$') solutions/*.ipynb; do \
	  jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=180 --output-dir "build/nb/$$(dirname "$$nb")" "$$nb" || exit 1; done

tf-validate:
	cd infra/terraform && terraform init -backend=false -input=false >/dev/null && terraform fmt -check -recursive && terraform validate

demo:
	agentsec demo
