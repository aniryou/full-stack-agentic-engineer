#!/usr/bin/env bash
# Create the Auth Manager 3-legged OAuth provider with gcloud — the imperative twin of
# infra/terraform/auth_provider.tf, for people who want to see the exact command from the docs
# (docs/sources.md, "Auth Manager"). Prefer Terraform; use this if you cannot run it.
#
# Usage:
#   PROJECT_ID=... LOCATION=us-central1 CLIENT_ID=... CLIENT_SECRET=... \
#   AUTHORIZATION_URL=https://idp.example/o/oauth2/auth TOKEN_URL=https://idp.example/o/oauth2/token \
#   infra/scripts/create_auth_provider.sh [PROVIDER_NAME]
#
# The provider's callback URL is deterministic and must be registered at the identity provider:
#   https://agentidentitycredentials.googleapis.com/v1/projects/PROJECT_ID/locations/LOCATION/authProviders/NAME/oauthcallback
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
LOCATION="${LOCATION:-us-central1}"
NAME="${1:-${AUTH_PROVIDER_NAME:-tickets-3lo}}"
CLIENT_ID="${CLIENT_ID:?set CLIENT_ID}"
CLIENT_SECRET="${CLIENT_SECRET:?set CLIENT_SECRET (never commit it; pass via env)}"
AUTHORIZATION_URL="${AUTHORIZATION_URL:?set AUTHORIZATION_URL}"
TOKEN_URL="${TOKEN_URL:?set TOKEN_URL}"

gcloud services enable agentidentity.googleapis.com agentidentitycredentials.googleapis.com --project="${PROJECT_ID}"

# Exactly the documented command shape.
gcloud agent-identity auth-providers create "${NAME}" \
  --project="${PROJECT_ID}" \
  --location="${LOCATION}" \
  --three-legged-oauth-client-id="${CLIENT_ID}" \
  --three-legged-oauth-client-secret="${CLIENT_SECRET}" \
  --three-legged-oauth-authorization-url="${AUTHORIZATION_URL}" \
  --three-legged-oauth-token-url="${TOKEN_URL}"

cat <<EOF

Created projects/${PROJECT_ID}/locations/${LOCATION}/authProviders/${NAME}
Register this redirect URI at your identity provider:
  https://agentidentitycredentials.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/authProviders/${NAME}/oauthcallback

Next: bind the agent with roles/agentidentity.user (infra/scripts/grant_agent_iam.sh <principal>).
VERIFY: whether your gcloud version exposes flags for allowed scopes / PKCE / default continue URI
        (Terraform sets allowed_scopes, enable_pkce = true and default_continue_uri; the documented
        create command above does not).
EOF
