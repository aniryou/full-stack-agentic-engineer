.PHONY: install test demo notebooks adk-demo api worker zip

install:
	pip install -e ".[dev,services]"

test:
	python -m pytest -q -p no:warnings

demo:
	python scripts/local_demo.py

adk-demo:
	pip install -e ".[adk]" && python examples/adk_agent_engine/run_local.py "durable agents" && python examples/adk_agent_engine/run_local.py approve

notebooks:            ## execute the worked notebooks headlessly
	python scripts/run_notebooks.py

api:                  ## local control-plane API (memory backend)
	LRA_BACKEND=memory uvicorn services.api.main:app --port 8080 --reload

worker:               ## local worker (memory backend)
	LRA_BACKEND=memory uvicorn services.worker.main:app --port 8081 --reload
