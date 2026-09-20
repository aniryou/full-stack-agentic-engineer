"""Run mistral_workflow.py with no Mistral account: a Temporal dev server on your machine.

Mistral Workflows is hybrid — Mistral hosts the orchestrator, your workers run wherever you
are. Locally we swap the hosted orchestrator for `temporal server start-dev` (the SDK
downloads it on first use) and register the one search attribute the SDK expects.
Everything else — the workflow, the activities, the signals — is unchanged.
"""

import logging
from contextlib import asynccontextmanager

logging.disable(logging.WARNING)

from temporalio.common import SearchAttributeKey                      # noqa: E402
from temporalio.testing import WorkflowEnvironment                    # noqa: E402

from mistralai.workflows.core.config.config import config             # noqa: E402
from mistralai.workflows.testing import create_test_worker            # noqa: E402

TASK_QUEUE = "local"


@asynccontextmanager
async def local_worker(workflow_classes, activities):
    """Yields a Temporal client with a worker running the given workflows/activities."""
    config.worker.deployment_name = TASK_QUEUE
    for flag in ("mistral_workflows_otel_traces_export", "mistral_workflows_otel_metrics_export", "mistral_workflows_otel_logs_export"):
        setattr(config.common, flag, False)                          # keep it offline
    async with await WorkflowEnvironment.start_local(
            search_attributes=[SearchAttributeKey.for_keyword("OtelTraceId")], dev_server_log_level="error") as env:
        async with create_test_worker(env, workflow_classes, activities=activities, task_queue=TASK_QUEUE):
            yield env.client


async def start(client, goal, execution_id, **params):
    """What `POST /v1/workflows/invoice-agent/execute` does on the hosted platform."""
    return await client.start_workflow("invoice-agent", {"goal": goal, **params}, id=execution_id, task_queue=TASK_QUEUE)
