
# Agent Gateway (opt-in): the *network* policy enforcement point in front of everything the
# agent talks to (primer §4.1, §8, §10). In agent-to-anywhere (egress) mode it:
#   * terminates mTLS with the agent's SPIFFE certificate and re-binds credentials with DPoP
#     ("double-bound" tokens, docs/sources.md) — a stolen token is useless off-box;
#   * asks IAP for an IAM decision per SPIFFE ID on every registered destination
#     (roles/iap.egressor on the Agent Registry entry, see agent_registry.tf);
#   * can hand request/response bodies to Model Armor ("semantic governance");
#   * parses MCP so policies can match tools/call and tool names.
# Limitations to remember: destinations must be registered in Agent Registry (<= 5,000 per
# gateway), the gateway itself is not covered by VPC Service Controls, destinations need
# publicly trusted CA certificates, and Gemini Enterprise supports egress mode only.

locals {
  gateway_count = var.enable_agent_gateway ? 1 : 0
}

resource "google_network_services_agent_gateway" "egress" {
  count = local.gateway_count

  project     = var.project_id
  name        = var.agent_gateway_name
  location    = var.region
  description = "agentsec egress gateway: mTLS + DPoP, IAP per SPIFFE ID, Model Armor on MCP traffic"
  labels      = local.labels

  # `protocols = ["MCP"]` is deprecated in google 8.x (the gateway governs all HTTP traffic,
  # MCP and A2A included, and parses MCP natively), so it is intentionally not set.

  google_managed {
    governed_access_path = "AGENT_TO_ANYWHERE"
  }

  # Only project-scoped registries are supported today. The provider example omits the API
  # version segment; VERIFY if the API starts requiring //agentregistry.googleapis.com/v1/... .
  registries = ["//agentregistry.googleapis.com/projects/${var.project_id}/locations/${var.region}"]

  depends_on = [google_project_service.apis, google_agent_registry_service.mcp]
}

# --- IAP request authorization (IAM per SPIFFE ID) --------------------------------------------
# Service Extensions callout to IAP. With the extension attached, every request through the
# gateway is authorised against the IAM policy of the registered destination. DRY_RUN lets you
# audit decisions before enforcing (var.agent_gateway_iap_dry_run).

resource "google_network_services_authz_extension" "iap" {
  count = local.gateway_count

  project     = var.project_id
  name        = "${var.agent_gateway_name}-iap"
  location    = var.region
  description = "IAP request authorization for the agentsec egress gateway"
  service     = "iap.googleapis.com"
  timeout     = "1s"
  fail_open   = false # a failed authz callout must deny, never allow

  metadata = merge(
    { iapPolicyVersion = "V2" },
    var.agent_gateway_iap_dry_run ? { iamEnforcementMode = "DRY_RUN" } : {},
  )

  depends_on = [google_project_service.apis]
}

resource "google_network_security_authz_policy" "iap" {
  count = local.gateway_count

  project        = var.project_id
  name           = "${var.agent_gateway_name}-iap"
  location       = var.region
  description    = "Delegate request authorization on the agentsec gateway to IAP"
  action         = "CUSTOM"
  policy_profile = "REQUEST_AUTHZ"

  target {
    resources = [google_network_services_agent_gateway.egress[0].id]
  }

  custom_provider {
    authz_extension {
      resources = [google_network_services_authz_extension.iap[0].id]
    }
  }
}

# --- Model Armor content authorization (egress screening) --------------------------------------
# The same strict template from model_armor.tf, applied by the gateway to MCP requests and
# responses. This is the "untrusted content boundary" enforced outside the agent process
# (primer §6.1): tool results are screened before the model ever sees them.

resource "google_network_services_authz_extension" "model_armor" {
  count = local.gateway_count

  project     = var.project_id
  name        = "${var.agent_gateway_name}-model-armor"
  location    = var.region
  description = "Model Armor content screening for the agentsec egress gateway"
  service     = "modelarmor.${local.model_armor_location}.rep.googleapis.com"
  timeout     = "1s"
  fail_open   = false

  metadata = {
    model_armor_settings = jsonencode([
      {
        request_template_id  = google_model_armor_template.strict.name
        response_template_id = google_model_armor_template.strict.name
      }
    ])
  }

  depends_on = [google_project_service.apis]
}

resource "google_network_security_authz_policy" "model_armor" {
  count = local.gateway_count

  project        = var.project_id
  name           = "${var.agent_gateway_name}-model-armor"
  location       = var.region
  description    = "Screen gateway traffic with Model Armor"
  action         = "CUSTOM"
  policy_profile = "CONTENT_AUTHZ"

  target {
    resources = [google_network_services_agent_gateway.egress[0].id]
  }

  custom_provider {
    authz_extension {
      resources = [google_network_services_authz_extension.model_armor[0].id]
    }
  }
}

# The gateway's Service Extensions service account must be allowed to call Model Armor with our
# template (docs: roles/modelarmor.calloutUser + serviceusage.serviceUsageConsumer in the gateway
# project, roles/modelarmor.user in the template project — here the same project).
resource "google_project_iam_member" "gateway_model_armor" {
  for_each = var.enable_agent_gateway ? toset([
    "roles/modelarmor.calloutUser",
    "roles/modelarmor.user",
    "roles/serviceusage.serviceUsageConsumer",
  ]) : toset([])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_network_services_agent_gateway.egress[0].agent_gateway_card[0].service_extensions_service_account}"
}
