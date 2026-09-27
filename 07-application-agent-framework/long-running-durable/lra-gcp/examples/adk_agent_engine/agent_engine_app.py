"""Deploy the ADK workflow to Vertex AI Agent Engine (managed runtime).

Agent Engine gives the workflow what our Cloud Run engine had to build by
hand: persistent sessions (``VertexAiSessionService``), scale-to-zero
between human-review pauses, autoscaling, and Cloud Trace integration.

Deploy (either):

    # 1. Agents CLI (scaffolds config + deploys)
    uv tool install google-agents-cli && agents-cli deploy

    # 2. Vertex AI SDK, from a Python shell / CI job
    python examples/adk_agent_engine/agent_engine_app.py --deploy

Then drive it over REST (`:query` / `:streamQuery`) or the SDK; the approval
webhook resumes a paused invocation exactly as run_local.py does, with a
FunctionResponse for ``REVIEW_CALL_ID``.

Verify the class/module paths against the current Vertex AI SDK before relying
on them in CI — Agent Engine's Python surface has moved between releases.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vertexai  # noqa: E402
from vertexai.agent_engines.templates.adk import AdkApp  # noqa: E402

from workflow_agent import app as adk_app  # noqa: E402


class ResearchAgentEngineApp(AdkApp):
    def set_up(self) -> None:  # runs once per replica at startup
        vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("LRA_LOCATION", "us-central1"))
        super().set_up()


def build_agent_engine_app() -> ResearchAgentEngineApp:
    """Constructing AdkApp resolves the GCP project, so build it lazily (keeps the module importable offline)."""
    return ResearchAgentEngineApp(app=adk_app, enable_tracing=True)


def deploy() -> None:
    from vertexai import agent_engines

    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("LRA_LOCATION", "us-central1")
    staging = os.environ.get("LRA_STAGING_BUCKET")  # gs://...
    vertexai.init(project=project, location=location, staging_bucket=staging)
    remote = agent_engines.create(
        agent_engine=build_agent_engine_app(),
        display_name="lra-research-workflow",
        description="Plan/research/critique/review/publish long-running workflow (ADK 2 Workflow)",
        requirements=["google-cloud-aiplatform[agent_engines]", "google-adk"],
        extra_packages=[str(Path(__file__).resolve().parent / "workflow_agent.py")],
        env_vars={"ADK_APP_NAME": adk_app.name},
        min_instances=0,  # sleep is free while runs wait on humans
    )
    print("deployed:", remote.resource_name)


if __name__ == "__main__" and "--deploy" in sys.argv:
    deploy()
