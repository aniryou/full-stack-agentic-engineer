#!/usr/bin/env bash
# Build the source archive Terraform inlines into google_vertex_ai_reasoning_engine
# (infra/terraform/reasoning_engine.tf, path B). Layout inside the tarball:
#
#   deploy/agent_engine_app.py   entrypoint_module = "deploy.agent_engine_app", entrypoint_object = "app"
#   agentsec/                    the package (copied from src/agentsec, no __pycache__)
#   policies/support-agent.yaml  AGENTSEC_POLICY
#   requirements-gcp.txt         requirements_file
#
# Usage: infra/scripts/build_agent_source.sh [OUTPUT]   (default infra/build/agent_source.tar.gz)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-${REPO_ROOT}/infra/build/agent_source.tar.gz}"
STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT

mkdir -p "${STAGE}/deploy" "${STAGE}/policies" "$(dirname "${OUT}")"
cp "${REPO_ROOT}"/infra/deploy/*.py "${STAGE}/deploy/"
cp "${REPO_ROOT}"/policies/*.yaml "${STAGE}/policies/"
cp "${REPO_ROOT}/infra/scripts/requirements-gcp.txt" "${STAGE}/requirements-gcp.txt"
# rsync-free copy of the package without caches / egg-info
(cd "${REPO_ROOT}/src" && find agentsec -type f -name '*.py' -not -path '*/__pycache__/*' | tar -cf - -T -) | tar -xf - -C "${STAGE}"

# Deterministic-ish tarball (sorted, no owner info) so re-running does not churn the plan.
# GNU tar has the reproducibility flags; fall back to plain tar (bsdtar on macOS).
if tar --version 2>/dev/null | grep -q 'GNU tar'; then
  tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='2026-01-01 00:00Z' \
    -C "${STAGE}" -czf "${OUT}" deploy agentsec policies requirements-gcp.txt
else
  tar -C "${STAGE}" -czf "${OUT}" deploy agentsec policies requirements-gcp.txt
fi

echo "wrote ${OUT} ($(du -h "${OUT}" | cut -f1))"
echo "next: terraform -chdir=infra/terraform apply -var deploy_agent_with_terraform=true -var agent_source_archive=${OUT}"
