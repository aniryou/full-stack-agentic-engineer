
# VPC Service Controls (opt-in): the data boundary (primer §8). A regular perimeter around the
# project restricts the services agents and tools touch, and Agent Identity principals are
# first-class in ingress/egress rules — including principalSet://<trust domain>/attribute.*
# forms (VPC-SC "Supported identities"). Combined with the MCP attributes (mcp.toolName,
# mcp.method, mcp.tool.isReadOnly) you get "agents in this project may only call read-only
# tools on that MCP server" as a perimeter rule; see infra/scripts/vpc_sc_mcp_rule.yaml for the
# attribute-conditioned variant, which is kept in gcloud YAML because the condition syntax
# should be checked against current docs before use.
#
# Requirements: an organisation-level access policy (var.access_policy_id) and
# roles/accesscontextmanager.policyAdmin. Adding aiplatform.googleapis.com as a restricted
# service also auto-blocks public internet access from Agent Platform (docs/sources.md).

locals {
  vpc_sc_count = var.enable_vpc_sc ? 1 : 0

  vpc_sc_restricted_services = [
    "aiplatform.googleapis.com",
    "secretmanager.googleapis.com",
    "run.googleapis.com",
    "agentidentity.googleapis.com",            # docs: add both Agent Identity APIs, use restricted VIP
    "agentidentitycredentials.googleapis.com", # (restricted.googleapis.com)
    "storage.googleapis.com",
    "bigquery.googleapis.com",
  ]
}

resource "google_access_context_manager_service_perimeter" "agents" {
  count = local.vpc_sc_count

  parent         = "accessPolicies/${var.access_policy_id}"
  name           = "accessPolicies/${var.access_policy_id}/servicePerimeters/agentsec_${var.project_number}"
  title          = "agentsec perimeter (${var.project_id})"
  description    = "Data boundary for the agentsec lab: agent runtime, secrets, MCP server, audit data"
  perimeter_type = "PERIMETER_TYPE_REGULAR"

  status {
    resources           = ["projects/${var.project_number}"]
    restricted_services = local.vpc_sc_restricted_services

    vpc_accessible_services {
      enable_restriction = true
      allowed_services   = ["RESTRICTED-SERVICES"]
    }
  }

  # Egress rules are managed by the standalone resource below (provider requirement).
  lifecycle {
    ignore_changes = [status[0].egress_policies]
  }

  depends_on = [google_project_service.apis]
}

# Egress rule: every agent of this project (principalSet) may call Cloud Run in the project that
# hosts the MCP server. Same project by default, in which case the rule is a no-op but shows the
# shape used for a separate "tools" project.
resource "google_access_context_manager_service_perimeter_egress_policy" "agents_to_mcp" {
  count = local.vpc_sc_count

  perimeter = google_access_context_manager_service_perimeter.agents[0].name
  title     = "agents-to-mcp-server"

  egress_from {
    # Agent principals in a perimeter rule. VERIFY: the provider's field description still
    # mentions user/group/serviceAccount only; the VPC-SC docs list agent identities and
    # principalSet://<trust domain>/attribute.* as supported in ingress/egress rules.
    identities = [local.all_project_agents]
  }

  egress_to {
    resources = ["projects/${local.vpc_sc_mcp_project}"]

    operations {
      service_name = "run.googleapis.com"
      method_selectors {
        method = "*"
      }
    }
  }

  lifecycle {
    create_before_destroy = true
  }
}
