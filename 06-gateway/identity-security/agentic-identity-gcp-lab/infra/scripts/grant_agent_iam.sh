#!/usr/bin/env bash
# Grant the agent principal everything it needs — and nothing more — with gcloud.
#
# Terraform (infra/terraform/iam.tf & friends) manages the same bindings except one: the
# provider (google 8.1.0) has no IAM resource for Auth Manager auth providers, so
# roles/agentidentity.user is granted here. The rest is repeated so this script is a complete,
# idempotent reference of "what an agent gets" (primer §4.5), e.g. for migrating an agent from a
# service account to Agent Identity: the new principal inherits nothing (docs/sources.md).
#
# Usage:
#   PROJECT_ID=... REGION=us-central1 infra/scripts/grant_agent_iam.sh "principal://agents.global.org-.../reasoningEngines/ID"
# Optional env:
#   AUTH_PROVIDER (default tickets-3lo)   AUTH_PROVIDER_LOCATION (default $REGION)
#   MCP_SERVICE   (default agentsec-mcp-tickets)   SECRET (default agentsec-demo-secret)
#   MCP_SERVER_ID (Agent Registry mcpServers ID; skip the IAP binding when unset)
#   MCP_ACCESS_ROLE (default roles/iap.egressor)
set -euo pipefail

PRINCIPAL="${1:?usage: grant_agent_iam.sh principal://...}"
case "${PRINCIPAL}" in
  principal://*|principalSet://*) ;;
  *) echo "member must be an IAM principal:// or principalSet:// identifier"; exit 2 ;;
esac

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"
AUTH_PROVIDER="${AUTH_PROVIDER:-tickets-3lo}"
AUTH_PROVIDER_LOCATION="${AUTH_PROVIDER_LOCATION:-${REGION}}"
MCP_SERVICE="${MCP_SERVICE:-agentsec-mcp-tickets}"
SECRET="${SECRET:-agentsec-demo-secret}"
MCP_ACCESS_ROLE="${MCP_ACCESS_ROLE:-roles/iap.egressor}"

echo ">> project-level platform roles (docs: recommended defaults for Agent Engine agents)"
for role in roles/aiplatform.expressUser roles/serviceusage.serviceUsageConsumer roles/browser roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" --member="${PRINCIPAL}" --role="${role}" --condition=None --quiet >/dev/null
  echo "   ${role}"
done

echo ">> Auth Manager: roles/agentidentity.user on the 3LO provider (documented gcloud form)"
gcloud agent-identity auth-providers add-iam-policy-binding "${AUTH_PROVIDER}" \
  --project="${PROJECT_ID}" \
  --location="${AUTH_PROVIDER_LOCATION}" \
  --role=roles/agentidentity.user \
  --member="${PRINCIPAL}"

echo ">> Secret Manager: accessor on the ONE demo secret"
gcloud secrets add-iam-policy-binding "${SECRET}" --project="${PROJECT_ID}" \
  --member="${PRINCIPAL}" --role=roles/secretmanager.secretAccessor --quiet >/dev/null

echo ">> Cloud Run: invoker on the MCP service only"
gcloud run services add-iam-policy-binding "${MCP_SERVICE}" --project="${PROJECT_ID}" --region="${REGION}" \
  --member="${PRINCIPAL}" --role=roles/run.invoker --quiet >/dev/null

if [[ -n "${MCP_SERVER_ID:-}" ]]; then
  echo ">> IAP: ${MCP_ACCESS_ROLE} on the registered MCP server (enforced at Agent Gateway)"
  # Documented form: gcloud iap web add-iam-policy-binding --resource-type=agent-registry --mcp-server=ID
  gcloud iap web add-iam-policy-binding \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --resource-type=agent-registry \
    --mcp-server="${MCP_SERVER_ID}" \
    --member="${PRINCIPAL}" \
    --role="${MCP_ACCESS_ROLE}"
else
  echo ">> IAP binding skipped (set MCP_SERVER_ID=<Agent Registry mcpServers ID>; Terraform output mcp_server_registry_id)"
fi

echo
echo "Done. Verify with:"
echo "  gcloud projects get-iam-policy ${PROJECT_ID} --flatten=bindings[].members --filter='bindings.members:${PRINCIPAL}' --format='table(bindings.role)'"
echo "  gcloud agent-identity auth-providers get-iam-policy ${AUTH_PROVIDER} --project=${PROJECT_ID} --location=${AUTH_PROVIDER_LOCATION}"
