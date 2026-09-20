.PHONY: install test notebooks build-notebooks run-local run-adk-web deploy clean

install:            ## editable install with dev extras
	pip install -e ".[dev]"

test:               ## run the offline test suite (~2s)
	python -m pytest tests -q

build-notebooks:    ## regenerate notebooks from tools/build_notebooks.py (clears outputs)
	python tools/build_notebooks.py

notebooks: build-notebooks  ## regenerate and execute the worked notebooks in place
	cd notebooks && for nb in 01_durable_loop_worked 02_fan_out_fan_in_worked 03_hitl_saga_scheduled_worked 04_adk_workflow_worked; do \
	  python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=240 $$nb.ipynb; done

run-local:          ## library service with the in-memory backend on :8080
	LRAGENTS_BACKEND=memory python -m lragents.service.app

run-adk-web:        ## ADK dev UI for the model-backed agent (needs Vertex creds)
	GOOGLE_GENAI_USE_VERTEXAI=1 adk web src/lragents --session_service_uri sqlite+aiosqlite:///sessions.db

deploy:             ## Cloud Run + Workflows (see infra/deploy.sh)
	bash infra/deploy.sh

clean:
	find . -name __pycache__ -o -name .pytest_cache -o -name .ipynb_checkpoints | xargs rm -rf
