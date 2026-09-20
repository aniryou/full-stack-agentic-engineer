
# Auth Manager: the broker for the agent's *delegated* authority (primer §3.2, §3.5).
# The agent never holds the CRM client secret or a refresh token — it asks Auth Manager for a
# user-scoped access token (RetrieveCredentials) and Auth Manager runs the 3-legged consent
# flow with the user. ADK plugs in through GcpAuthProvider / GcpAuthProviderScheme.

resource "google_agent_identity_auth_provider" "crm" {
  project          = var.project_id
  location         = local.auth_provider_location
  auth_provider_id = var.auth_provider_id
  description      = "3LO provider for the Acme CRM (user-delegated customer lookups)"
  labels           = local.labels

  # Scope minimisation is enforced by the broker, not by the tool code.
  allowed_scopes = var.crm_allowed_scopes

  # The resource also accepts `workload_ids` (input-only list of principal:// agents that will
  # use the provider). It is intentionally NOT set here: the reasoning engine's env references
  # this provider, so referencing the engine back from here would be a dependency cycle, and
  # VERIFY: whether workload_ids creates the roles/agentidentity.user binding at all. The
  # explicit binding is applied by infra/scripts/grant_agent_iam.sh.

  auth_provider_type_params {
    three_legged_oauth {
      client_id                = var.crm_oauth_client_id
      client_secret_wo         = var.crm_oauth_client_secret # write-only: not stored in state
      client_secret_wo_version = tostring(var.crm_oauth_client_secret_version)
      authorization_url        = var.crm_authorization_url
      token_url                = var.crm_token_url
      enable_pkce              = true # PKCE (S256) is mandatory in OAuth 2.1 / MCP auth (primer §7.1)
      default_continue_uri     = var.crm_continue_uri
    }
  }

  depends_on = [google_project_service.apis]
}

# IAM on the provider: the agent principal needs roles/agentidentity.user to call
# retrieveCredentials for this provider. The google provider (8.1.0) has no
# google_agent_identity_auth_provider_iam_* resource, so the binding is applied with gcloud:
#
#   gcloud agent-identity auth-providers add-iam-policy-binding <ID> \
#     --project=<PROJECT> --location=<LOCATION> \
#     --role=roles/agentidentity.user --member=principal://...
#
# See infra/scripts/grant_agent_iam.sh (takes the principal as $1).
