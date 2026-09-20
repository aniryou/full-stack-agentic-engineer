
# IAM for the agent principal. Three ideas from the primer are visible in this file:
#   * least privilege by *resource*, not by project: the agent gets project-wide roles only
#     where the platform needs them (expressUser/serviceUsageConsumer/browser/logWriter) and
#     resource-scoped roles everywhere else (one secret, one Cloud Run service, one auth
#     provider, one registry entry) — primer §4.5;
#   * fleet-level policy with principalSet: things that must be true for *every* agent in the
#     project are attached to the principalSet, not copied per agent;
#   * the envelope: a deny policy (and optionally a PAB policy) bounds what any allow-policy can
#     ever grant an agent — the "never list".
#
# Bindings that Terraform cannot express (no provider resource) live in
# infra/scripts/grant_agent_iam.sh: roles/agentidentity.user on the Auth Manager provider.

# --- Project-level roles recommended for Agent Engine agents (docs/sources.md) ---------------
# The Agent Identity guide recommends aiplatform.expressUser; the Auth Manager (3LO) guide lists
# aiplatform.user. expressUser is the narrower of the two and is what the lab starts with —
# widen only if a Vertex feature the agent uses (e.g. extensions) demands aiplatform.user.
resource "google_project_iam_member" "agent_platform_roles" {
  for_each = local.agent_identity_known ? toset([
    "roles/aiplatform.expressUser",            # call Gemini as itself
    "roles/serviceusage.serviceUsageConsumer", # consume APIs against the project quota
    "roles/browser",                           # resolve project metadata
    "roles/logging.logWriter",                 # write the agentsec-audit log
  ]) : toset([])

  project = var.project_id
  role    = each.value
  member  = local.agent_principal
}

# (Resource-scoped grants for the same principal are next to the resources they scope:
#   secrets.tf        -> roles/secretmanager.secretAccessor on the demo secret only
#   cloudrun_mcp.tf   -> roles/run.invoker on the MCP service only
#   agent_registry.tf -> roles/iap.egressor on the registered MCP server only
#   auth_provider.tf  -> roles/agentidentity.user via gcloud)

# --- Deny policy: the never-list for every agent in the project ------------------------------
# Deny policies are evaluated before allow policies: even a mistaken roles/owner grant to an
# agent cannot delete objects or mint service-account keys. Permissions use the
# "<service>.googleapis.com/<resource>.<verb>" deny-policy format.
resource "google_iam_deny_policy" "agents_never" {
  count = var.enable_agent_deny_policy ? 1 : 0

  parent       = urlencode("cloudresourcemanager.googleapis.com/projects/${var.project_id}")
  name         = "agentsec-agents-never-list"
  display_name = "Agents in this project may never delete objects or create SA keys"

  rules {
    description = "Rogue-agent containment (OWASP ASI10) and no long-lived keys (primer §5)"
    deny_rule {
      # VERIFY: agent principalSet members are documented for allow/deny/PAB/VPC-SC policies;
      # confirm the deny-policy API accepts this exact principalSet form.
      denied_principals = [local.all_project_agents]
      denied_permissions = [
        "storage.googleapis.com/objects.delete",
        "iam.googleapis.com/serviceAccountKeys.create",
      ]
    }
  }

  depends_on = [google_project_service.apis]
}

# --- Principal Access Boundary (opt-in, organisation-level) ----------------------------------
# A PAB policy limits the *resources* a principal can access at all, independent of any allow
# policy: principals bound to it can only use resources inside this project. Requires the
# org-level PAB admin role; bound at the project so it stays scoped to this lab.
resource "google_iam_principal_access_boundary_policy" "project_only" {
  count = var.enable_pab && var.org_id != null ? 1 : 0

  organization                        = var.org_id
  location                            = "global"
  principal_access_boundary_policy_id = "agentsec-${var.project_number}-project-only"
  display_name                        = "agentsec: confine principals to project ${var.project_id}"

  details {
    rules {
      description = "Only resources under this project are eligible"
      effect      = "ALLOW"
      resources   = ["//cloudresourcemanager.googleapis.com/projects/${var.project_id}"]
    }
    enforcement_version = "latest"
  }

  depends_on = [google_project_service.apis]
}

# Bind it to the project's principal set. VERIFY: whether Agent Identity principals of the
# project are members of the project principal set for PAB purposes, or whether the binding
# target must be the agents' trust domain — the supported target formats documented for the
# binding are projects, folders, organisations and workload identity pools.
resource "google_iam_projects_policy_binding" "pab" {
  count = var.enable_pab && var.org_id != null ? 1 : 0

  project           = var.project_id
  location          = "global"
  policy_binding_id = "agentsec-pab-binding"
  display_name      = "agentsec PAB binding"
  policy_kind       = "PRINCIPAL_ACCESS_BOUNDARY"
  policy            = "organizations/${var.org_id}/locations/global/principalAccessBoundaryPolicies/${google_iam_principal_access_boundary_policy.project_only[0].principal_access_boundary_policy_id}"

  target {
    principal_set = "//cloudresourcemanager.googleapis.com/projects/${var.project_id}"
  }

  # The PAB policy takes a moment to become bindable after creation; re-run apply if the first
  # attempt returns NOT_FOUND (the provider docs use a time_sleep here).
  depends_on = [google_iam_principal_access_boundary_policy.project_only]
}
