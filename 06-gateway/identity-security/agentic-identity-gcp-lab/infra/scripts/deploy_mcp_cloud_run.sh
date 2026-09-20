#!/usr/bin/env bash
# Deploy (or re-deploy) the tickets MCP server on Cloud Run *as an MCP server with its own
# Agent Identity*. Terraform (infra/terraform/cloudrun_mcp.tf) creates the service with a
# fallback service account because `--functional-type` / `--identity-type` are beta gcloud
# flags with no Terraform equivalent; this script flips it to Agent Identity, which
#   * gives the service a SPIFFE identity (principal://<trust domain>/resources/run/...),
#   * auto-registers it in Agent Registry under /mcpServers,
#   * keeps the service private (--no-allow-unauthenticated, --ingress=internal).
#
# Usage:
#   PROJECT_ID=my-project REGION=us-central1 infra/scripts/deploy_mcp_cloud_run.sh [IMAGE]
# Env:
#   PROJECT_ID (required), REGION (default us-central1), SERVICE (default agentsec-mcp-tickets)
#   IMAGE      (default <REGION>-docker.pkg.dev/<PROJECT_ID>/agentsec/mcp-tickets:latest)
#   BUILD=1    build+push IMAGE from the repo root with Cloud Build first (uses Dockerfile.mcp)
#   IDENTITY_TYPE (default agent-identity; set service-account to keep the fallback SA)
#   INGRESS    (default internal; `all` for an IAM-gated smoke test from your laptop)
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-agentsec-mcp-tickets}"
IMAGE="${1:-${IMAGE:-${REGION}-docker.pkg.dev/${PROJECT_ID}/agentsec/mcp-tickets:latest}}"
IDENTITY_TYPE="${IDENTITY_TYPE:-agent-identity}"
INGRESS="${INGRESS:-internal}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
MCP_URL="https://${SERVICE}-${PROJECT_NUMBER}.${REGION}.run.app/mcp" # Cloud Run deterministic URL

if [[ "${BUILD:-0}" == "1" ]]; then
  echo ">> building ${IMAGE} with Cloud Build (infra/scripts/Dockerfile.mcp)"
  # Cloud Build needs the Dockerfile at the context root; stage a copy so the image build
  # sees the whole package (src/, pyproject.toml).
  cp "${REPO_ROOT}/infra/scripts/Dockerfile.mcp" "${REPO_ROOT}/Dockerfile"
  trap 'rm -f "${REPO_ROOT}/Dockerfile"' EXIT
  gcloud builds submit "${REPO_ROOT}" --project="${PROJECT_ID}" --region="${REGION}" --tag="${IMAGE}"
fi

echo ">> deploying ${SERVICE} in ${REGION} as functional-type=mcp-server, identity-type=${IDENTITY_TYPE}"
# Exactly the documented form (docs/sources.md, Cloud Run "Agent Platform features"):
#   gcloud beta run deploy SERVICE --image=... --functional-type=mcp-server --identity-type=agent-identity
gcloud beta run deploy "${SERVICE}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${IMAGE}" \
  --functional-type=mcp-server \
  --identity-type="${IDENTITY_TYPE}" \
  --no-allow-unauthenticated \
  --ingress="${INGRESS}" \
  --port=8080 \
  --set-env-vars="AGENTSEC_PROFILE=gcp,AGENTSEC_PROJECT_ID=${PROJECT_ID},AGENTSEC_PROJECT_NUMBER=${PROJECT_NUMBER},AGENTSEC_LOCATION=${REGION},AGENTSEC_MCP_URL=${MCP_URL},AGENTSEC_AUDIT_LOG=${AUDIT_LOG:-agentsec-audit}"

echo ">> identity of the latest revision (look for the agent identity / SPIFFE ID):"
REVISION="$(gcloud run services describe "${SERVICE}" --project="${PROJECT_ID}" --region="${REGION}" --format='value(status.latestReadyRevisionName)')"
gcloud beta run revisions describe "${REVISION}" --project="${PROJECT_ID}" --region="${REGION}" --format=yaml | grep -iE 'identity|spiffe|serviceAccount' || true

echo ">> registry entries for this project (auto-registered MCP servers):"
gcloud agent-registry mcp-servers list --project="${PROJECT_ID}" --location="${REGION}" 2>/dev/null || \
  echo "   (gcloud agent-registry not available in this SDK version — check the Agent Registry page in the console)"

echo
echo "MCP endpoint: ${MCP_URL}"
echo "Remember: the service's Agent Identity is a NEW principal — nothing granted to the fallback SA carries over."
