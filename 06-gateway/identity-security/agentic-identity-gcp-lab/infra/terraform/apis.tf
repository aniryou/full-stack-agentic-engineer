
# Service enablement. Everything the agent, the MCP server and the governance plane need.
# Opt-in layers only enable their API when the corresponding flag is on, so a plain apply in a
# project without org-level permissions does not fail.

locals {
  base_apis = [
    "aiplatform.googleapis.com",               # Agent Engine (reasoningEngines) + Gemini
    "run.googleapis.com",                      # MCP server hosting
    "iam.googleapis.com",                      # service accounts, WIF, deny policies
    "iamcredentials.googleapis.com",           # short-lived SA impersonation (no keys)
    "sts.googleapis.com",                      # Workload Identity Federation token exchange
    "secretmanager.googleapis.com",            # the one demo secret
    "modelarmor.googleapis.com",               # prompt/response screening
    "logging.googleapis.com",                  # audit sink
    "bigquery.googleapis.com",                 # audit destination
    "artifactregistry.googleapis.com",         # MCP server image
    "cloudbuild.googleapis.com",               # image builds from CI
    "networkservices.googleapis.com",          # Agent Gateway (agentGateways), authz extensions
    "iap.googleapis.com",                      # IAM enforcement per SPIFFE ID at the gateway
    "agentidentity.googleapis.com",            # Agent Identity + Auth Manager control plane
    "agentidentitycredentials.googleapis.com", # retrieveCredentials / oauthcallback data plane
    "agentregistry.googleapis.com",            # Agent Registry (agents, mcpServers, endpoints)
    "cloudresourcemanager.googleapis.com",
    "serviceusage.googleapis.com",
    "storage.googleapis.com", # Agent Engine staging bucket
  ]

  optional_apis = concat(
    var.enable_vpc_sc ? ["accesscontextmanager.googleapis.com"] : [],
    var.enable_org_policy ? ["orgpolicy.googleapis.com"] : [],
    var.enable_agent_gateway ? ["networksecurity.googleapis.com", "compute.googleapis.com", "dns.googleapis.com"] : [],
  )
}

resource "google_project_service" "apis" {
  for_each = toset(concat(local.base_apis, local.optional_apis))

  project                    = var.project_id
  service                    = each.value
  disable_on_destroy         = false
  disable_dependent_services = false
}
