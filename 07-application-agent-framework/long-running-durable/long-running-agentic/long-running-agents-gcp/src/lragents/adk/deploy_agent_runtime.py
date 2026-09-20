"""Deploy the ADK agent to Agent Runtime (Gemini Enterprise Agent Platform,
formerly Vertex AI Agent Engine) with managed Sessions + Memory Bank.

    pip install "google-cloud-aiplatform[agent_engines,adk]"
    python -m lragents.adk.deploy_agent_runtime

Trade-off vs Cloud Run (see docs/00_primer.md §6): Agent Runtime gives you
managed sessions, memory bank, scaling and a query API with no container to
own; but you drive wake-ups from *outside* (Cloud Scheduler → its query
endpoint) because the ADK trigger routes only exist when you host the FastAPI
app yourself.
"""

from __future__ import annotations

import os


def deploy() -> str:
    import vertexai
    from vertexai.agent_engines import AdkApp

    from .agent import root_agent

    client = vertexai.Client(project=os.environ["GOOGLE_CLOUD_PROJECT"], location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
    adk_app = AdkApp(agent=root_agent)
    engine = client.agent_engines.create(
        agent_engine=adk_app,
        config={
            "staging_bucket": os.environ["STAGING_BUCKET"],           # gs://...
            "requirements": ["google-cloud-aiplatform[agent_engines,adk]"],
            "display_name": "concert-long-running",
        },
    )
    print("deployed:", engine.api_resource.name)
    return engine.api_resource.name


if __name__ == "__main__":
    deploy()
