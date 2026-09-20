
# Org Policy custom constraints (opt-in, organisation-level): guardrails on *how agents may be
# created*, evaluated by the control plane before a resource exists (primer §4.5). They complement
# IAM (who may act) with "what shapes of resources are acceptable at all".
#
# Requires var.org_id and roles/orgpolicy.policyAdmin on the organisation.

locals {
  org_policy_count = var.enable_org_policy && var.org_id != null ? 1 : 0
}

# ------------------------------------------------------------------------------------------
# 1. Auth Manager providers must use PKCE for 3-legged OAuth.
#    Resource type agentidentity.googleapis.com/AuthProvider and the field
#    resource.authProviderTypeParams.threeLeggedOauth.enablePkce come from the documented
#    custom-constraint reference for Auth Manager (docs.cloud.google.com/iam/docs/
#    agent-identity-custom-constraints) — the example there is used verbatim.
# ------------------------------------------------------------------------------------------
resource "google_org_policy_custom_constraint" "require_pkce_for_auth_providers" {
  count = local.org_policy_count

  name         = "custom.requirePkceForAgentAuthProviders"
  parent       = "organizations/${var.org_id}"
  display_name = "Auth Manager 3LO providers must enable PKCE"
  description  = "Three-legged OAuth providers created without PKCE are rejected (OAuth 2.1 / MCP authorization baseline)."

  action_type    = "DENY"
  condition      = "has(resource.authProviderTypeParams.threeLeggedOauth) && resource.authProviderTypeParams.threeLeggedOauth.enablePkce == false"
  method_types   = ["CREATE", "UPDATE"]
  resource_types = ["agentidentity.googleapis.com/AuthProvider"]
}

resource "google_org_policy_policy" "require_pkce_for_auth_providers" {
  count = local.org_policy_count

  name   = "projects/${var.project_id}/policies/${google_org_policy_custom_constraint.require_pkce_for_auth_providers[0].name}"
  parent = "projects/${var.project_id}"

  spec {
    rules {
      enforce = "TRUE"
    }
  }
}

# ------------------------------------------------------------------------------------------
# 2. Reasoning engines must run with Agent Identity.
#
#    #########################################################################################
#    # VERIFY — ILLUSTRATIVE: `aiplatform.googleapis.com/ReasoningEngine` is NOT listed on the #
#    # custom-constraint "supported services" page as of the research date, and the field     #
#    # path `resource.spec.identityType` is inferred from the REST resource shape. Check the   #
#    # current reference before enabling; the API will reject an unsupported resource type.   #
#    #########################################################################################
# ------------------------------------------------------------------------------------------
resource "google_org_policy_custom_constraint" "require_agent_identity" {
  count = local.org_policy_count

  name         = "custom.requireAgentIdentityForReasoningEngines"
  parent       = "organizations/${var.org_id}"
  display_name = "Agent Engine deployments must use Agent Identity"
  description  = "Reasoning engines may only be created or updated with spec.identityType = AGENT_IDENTITY (no shared service accounts)."

  action_type    = "ALLOW"
  condition      = "resource.spec.identityType == 'AGENT_IDENTITY'"
  method_types   = ["CREATE", "UPDATE"]
  resource_types = ["aiplatform.googleapis.com/ReasoningEngine"]
}

resource "google_org_policy_policy" "require_agent_identity" {
  count = local.org_policy_count

  name   = "projects/${var.project_id}/policies/${google_org_policy_custom_constraint.require_agent_identity[0].name}"
  parent = "projects/${var.project_id}"

  spec {
    rules {
      enforce = "TRUE"
    }
  }
}
