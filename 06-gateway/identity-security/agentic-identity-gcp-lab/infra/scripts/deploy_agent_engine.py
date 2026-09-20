#!/usr/bin/env python3
"""Deploy the agentsec support agent to Agent Engine *with Agent Identity* (SDK path A).

This is the deployment shape from the Google docs (docs/sources.md, "Deploy with identity"):

    client = vertexai.Client(project=..., location=..., http_options=dict(api_version="v1beta1"))
    client.agent_engines.create(
        agent=AdkApp(agent=...),
        config={
            "display_name": ...,
            "identity_type": types.IdentityType.AGENT_IDENTITY,
            "requirements": ["google-cloud-aiplatform[agent_engines,adk]", "google-adk[agent-identity,mcp]>=2.8.0"],
            "staging_bucket": f"gs://{bucket}",
            "env_vars": {...},
        },
    )

Why a script and not Terraform by default: the SDK packages the ADK app, uploads it to the
staging bucket and registers the ADK method surface for you; Terraform (reasoning_engine.tf,
opt-in) needs a source archive and a hand-written class_methods declaration instead.

Usage (after `terraform apply`):
    python infra/scripts/deploy_agent_engine.py --project P --location us-central1 \
        --staging-bucket P-agentsec-staging --env-from-terraform infra/terraform
    # then: terraform apply -var agent_engine_id=<printed ID>

The AGENTSEC_* environment is read on THIS machine when the app is built (the AdkApp is
pickled), so the same values are also passed as `env_vars` for the runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = REPO_ROOT / "infra" / "deploy"

REQUIREMENTS = [
    "google-cloud-aiplatform[agent_engines,adk]",
    "google-adk[agent-identity,mcp]>=2.8.0",
    "mcp>=1.27,<2",
    "google-cloud-modelarmor>=0.2",
    "google-cloud-logging>=3.10",
    "PyJWT[crypto]>=2.9",
    "cryptography>=43",
    "PyYAML>=6.0",
    "httpx>=0.27",
]


def _terraform_output(tf_dir: Path, name: str) -> dict[str, str]:
    cmd = ["terraform", f"-chdir={tf_dir}", "output", "-json", name]  # noqa: S607 - local tooling
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout  # noqa: S603
    value = json.loads(out)
    return {str(k): str(v) for k, v in value.items()}


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--project", default=os.environ.get("AGENTSEC_PROJECT_ID"), required=False)
    p.add_argument("--location", default=os.environ.get("AGENTSEC_LOCATION", "us-central1"))
    p.add_argument("--display-name", default="agentsec-support-agent")
    p.add_argument(
        "--staging-bucket", help="bucket name without gs:// (Terraform output agent_staging_bucket)"
    )
    p.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="runtime env var (repeatable)",
    )
    p.add_argument(
        "--env-from-terraform",
        metavar="TF_DIR",
        help="read `terraform output agent_env` from this directory",
    )
    p.add_argument(
        "--agent-gateway", help="projects/P/locations/L/agentGateways/NAME to route egress through"
    )
    p.add_argument(
        "--delete",
        metavar="ENGINE_NAME",
        help="delete an engine (projects/.../reasoningEngines/ID) and exit",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="build the app and print the config without deploying",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if not args.project:
        print("--project (or AGENTSEC_PROJECT_ID) is required", file=sys.stderr)
        return 2

    import vertexai
    from vertexai import types

    # v1beta1 is where identity_type / agent_gateway_config live (docs/sources.md).
    client = vertexai.Client(
        project=args.project, location=args.location, http_options=dict(api_version="v1beta1")
    )

    if args.delete:
        client.agent_engines.delete(name=args.delete, force=True)
        print(f"deleted {args.delete}")
        return 0

    if not args.staging_bucket:
        print("--staging-bucket is required to deploy", file=sys.stderr)
        return 2

    # Runtime environment: Terraform's local.agent_env, then explicit --env overrides.
    env_vars: dict[str, str] = {}
    if args.env_from_terraform:
        env_vars.update(_terraform_output(Path(args.env_from_terraform), "agent_env"))
    for item in args.env:
        key, _, value = item.partition("=")
        env_vars[key] = value
    env_vars.setdefault("AGENTSEC_PROFILE", "gcp")
    env_vars.setdefault("AGENTSEC_PROJECT_ID", args.project)
    env_vars.setdefault("AGENTSEC_LOCATION", args.location)
    # Mirror into this process so infra/deploy/agent_engine_app.py builds the same agent.
    os.environ.update(env_vars)

    sys.path.insert(0, str(REPO_ROOT / "infra"))
    from deploy.agent_engine_app import app  # noqa: E402  (builds the AdkApp + SecurityPlugin)

    config: dict[str, object] = {
        "display_name": args.display_name,
        "description": "agentsec reference support agent (ADK) with Agent Identity",
        # THE setting: the agent runs as its own SPIFFE principal, not as a service account.
        "identity_type": types.IdentityType.AGENT_IDENTITY,
        "requirements": REQUIREMENTS,
        "staging_bucket": f"gs://{args.staging_bucket}",
        "env_vars": env_vars,
        # Ship the package, the entrypoint module and the policy file alongside the pickle so
        # imports resolve in the runtime.
        "extra_packages": [
            str(REPO_ROOT / "src" / "agentsec"),
            str(DEPLOY_DIR),
            str(REPO_ROOT / "policies"),
        ],
    }
    if args.agent_gateway:
        # Same keys as the REST field spec.deploymentSpec.agentGatewayConfig.agentToAnywhereConfig
        config["agent_gateway_config"] = {
            "agent_to_anywhere_config": {"agent_gateway": args.agent_gateway}
        }

    if args.dry_run:
        printable = {k: (str(v) if k == "identity_type" else v) for k, v in config.items()}
        print(json.dumps(printable, indent=2, default=str))
        return 0

    remote = client.agent_engines.create(agent=app, config=config)
    resource = getattr(remote, "api_resource", remote)
    name = getattr(resource, "name", None) or str(remote)
    engine_id = name.rsplit("/", 1)[-1]
    spec = getattr(resource, "spec", None)
    effective_identity = getattr(spec, "effective_identity", None)

    print("deployed:", name)
    print("engine id:", engine_id)
    print(
        "effective identity:",
        effective_identity or "(check the Deployments page / REST effectiveIdentity)",
    )
    print()
    print("next:")
    print(f"  terraform -chdir=infra/terraform apply -var agent_engine_id={engine_id}")
    print(
        '  infra/scripts/grant_agent_iam.sh "$(terraform -chdir=infra/terraform output -raw agent_principal)"'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
