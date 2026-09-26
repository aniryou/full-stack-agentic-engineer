.PHONY: install test demo notebooks build-notebooks adk-demo api worker

install:
	pip install -e ".[dev,services]"

test:
	python -m pytest -q -p no:warnings

demo:
	python scripts/local_demo.py

adk-demo:             ## ADK 2: pause at the review gate, resume from a new process; then the ticket-queue graph
	pip install -e ".[adk]" && python examples/adk_agent_engine/run_local.py "durable agents" && python examples/adk_agent_engine/run_local.py approve \
	  && python examples/adk_ticket_queue/nightly_workflow.py

build-notebooks:      ## regenerate notebooks/ (blanks) and solutions/ (answers) from scripts/build_notebooks.py
	python scripts/build_notebooks.py

notebooks:            ## the solutions run clean; each blank stops at its first exercise (04 skips without the adk extra)
	python scripts/run_notebooks.py solutions && python scripts/run_notebooks.py notebooks --expect-fail

api:                  ## local control-plane API (memory backend)
	LRA_BACKEND=memory uvicorn services.api.main:app --port 8080 --reload

worker:               ## local worker (memory backend)
	LRA_BACKEND=memory uvicorn services.worker.main:app --port 8081 --reload
