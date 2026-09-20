
# ------------------------------------------------------------------------------------------
# Project / organisation
# ------------------------------------------------------------------------------------------

variable "project_id" {
  description = "Project ID that hosts the agent, the MCP server and the control-plane resources."
  type        = string
}

variable "project_number" {
  description = "Numeric project number. Agent Identity resource paths and principalSets are keyed by project *number*, not ID (primer §3.3)."
  type        = string

  validation {
    condition     = can(regex("^[0-9]+$", var.project_number))
    error_message = "project_number must be the numeric project number."
  }
}

variable "org_id" {
  description = "Numeric organisation ID. When set, the Agent Identity trust domain is agents.global.org-<ORG_ID>.system.id.goog; when null the project-scoped trust domain agents.global.project-<PROJECT_NUMBER>.system.id.goog is used. Org-level opt-ins (org policy, PAB) require it."
  type        = string
  default     = null
}

variable "region" {
  description = "Region for Agent Engine, Cloud Run, Agent Registry, Agent Gateway and the Auth Manager provider."
  type        = string
  default     = "us-central1"
}

variable "labels" {
  description = "Labels applied to every labelable resource."
  type        = map(string)
  default     = { app = "agentsec-lab" }
}

# ------------------------------------------------------------------------------------------
# Agent (Agent Engine / reasoning engine)
# ------------------------------------------------------------------------------------------

variable "agent_display_name" {
  description = "Display name of the Agent Engine deployment."
  type        = string
  default     = "agentsec-support-agent"
}

variable "deploy_agent_with_terraform" {
  description = <<-EOT
    When true, Terraform creates the google_vertex_ai_reasoning_engine from a source archive
    (var.agent_source_archive). When false (default) the agent is deployed with
    infra/scripts/deploy_agent_engine.py (Python SDK) and you pass the resulting engine ID in
    var.agent_engine_id so IAM bindings can be computed. See README.md for the trade-off.
  EOT
  type        = bool
  default     = false
}

variable "agent_engine_id" {
  description = "Existing reasoning-engine ID (the numeric last path segment of projects/*/locations/*/reasoningEngines/<ID>). Required for the agent IAM bindings when deploy_agent_with_terraform = false."
  type        = string
  default     = null
}

variable "agent_source_archive" {
  description = "Path to the .tar.gz produced by infra/scripts/build_agent_source.sh (only read when deploy_agent_with_terraform = true)."
  type        = string
  default     = "../build/agent_source.tar.gz"
}

variable "agent_model" {
  description = "Gemini model the support agent uses (AGENTSEC_MODEL)."
  type        = string
  default     = "gemini-2.5-flash"
}

variable "agent_max_instances" {
  description = "Upper bound for Agent Engine instances (cost control for a lab)."
  type        = number
  default     = 2
}

variable "kms_key_name" {
  description = "Optional CMEK key (projects/*/locations/<region>/keyRings/*/cryptoKeys/*) for the reasoning engine. Must be in var.region."
  type        = string
  default     = null
}

# ------------------------------------------------------------------------------------------
# MCP server on Cloud Run
# ------------------------------------------------------------------------------------------

variable "mcp_service_name" {
  description = "Cloud Run service name for the tickets MCP server."
  type        = string
  default     = "agentsec-mcp-tickets"
}

variable "mcp_server_image" {
  description = "Container image for the MCP server. Defaults to <region>-docker.pkg.dev/<project>/agentsec/mcp-tickets:latest (the Artifact Registry repo created in identity.tf)."
  type        = string
  default     = null
}

variable "mcp_ingress" {
  description = "Cloud Run ingress for the MCP server. INGRESS_TRAFFIC_INTERNAL_ONLY is the secure default (reach it via Agent Gateway / PSC-I). Use INGRESS_TRAFFIC_ALL only for a quick lab test — IAM invoker is still enforced."
  type        = string
  default     = "INGRESS_TRAFFIC_INTERNAL_ONLY"

  validation {
    condition     = contains(["INGRESS_TRAFFIC_ALL", "INGRESS_TRAFFIC_INTERNAL_ONLY", "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"], var.mcp_ingress)
    error_message = "mcp_ingress must be one of the Cloud Run v2 ingress enum values."
  }
}

variable "mcp_sts_issuer" {
  description = "Issuer URL the MCP server accepts tokens from (AGENTSEC_STS_ISSUER). In the gcp profile this is the authorization server in front of the MCP resource server (primer §7.1)."
  type        = string
  default     = "https://sts.agentsec.local"
}

variable "mcp_access_role" {
  description = "IAP role granted to the agent principal on the registered MCP server; enforced by Agent Gateway. The provider docs and the Agent Platform IAM guide use roles/iap.egressor (permission iap.resources.egressViaIAP)."
  type        = string
  default     = "roles/iap.egressor" # VERIFY: confirm against the current 'Configure IAM agent policies' page before relying on it.
}

# ------------------------------------------------------------------------------------------
# Auth Manager (3-legged OAuth provider for the CRM tool)
# ------------------------------------------------------------------------------------------

variable "auth_provider_id" {
  description = "Short ID of the Auth Manager provider (Settings.auth_provider_short in src/agentsec/config.py)."
  type        = string
  default     = "tickets-3lo"
}

variable "auth_provider_location" {
  description = "Location of the Auth Manager provider. Auth providers are regional in the docs (us-west1 in Google's example); defaults to var.region."
  type        = string
  default     = null
}

variable "crm_oauth_client_id" {
  description = "OAuth client ID registered at the CRM identity provider for the agent's delegated (3LO) access."
  type        = string
  default     = "acme-support-agent"
}

variable "crm_oauth_client_secret" {
  description = "OAuth client secret for the CRM 3LO client. Written with a write-only argument: never stored in state."
  type        = string
  sensitive   = true
  default     = null
}

variable "crm_oauth_client_secret_version" {
  description = "Bump this integer whenever crm_oauth_client_secret changes (write-only arguments need an explicit trigger)."
  type        = number
  default     = 1
}

variable "crm_authorization_url" {
  description = "Authorization endpoint of the CRM's OAuth server."
  type        = string
  default     = "https://idp.acme.example/o/oauth2/auth"
}

variable "crm_token_url" {
  description = "Token endpoint of the CRM's OAuth server."
  type        = string
  default     = "https://idp.acme.example/o/oauth2/token"
}

variable "crm_allowed_scopes" {
  description = "Scopes the agent may request through the provider (scope minimisation, primer §3.2). Matches CRM_SCOPE in src/agentsec/agents/root_agent.py."
  type        = list(string)
  default     = ["https://crm.acme.example/auth/customers.read"]
}

variable "crm_continue_uri" {
  description = "Front-end URL Auth Manager redirects to after consent (the /validateUserId endpoint, primer §3.5)."
  type        = string
  default     = "https://app.acme.example/validateUserId"
}

# ------------------------------------------------------------------------------------------
# Secrets
# ------------------------------------------------------------------------------------------

variable "demo_secret_value" {
  description = "Value of the single demo secret injected as AGENTSEC_SECRET_DEMO. Written with secret_data_wo so it never enters state."
  type        = string
  sensitive   = true
  default     = null
}

variable "demo_secret_version" {
  description = "Bump when demo_secret_value changes."
  type        = number
  default     = 1
}

# ------------------------------------------------------------------------------------------
# Model Armor
# ------------------------------------------------------------------------------------------

variable "model_armor_template_id" {
  description = "ID of the Model Armor template used by ModelArmorScreener and (optionally) by the Agent Gateway."
  type        = string
  default     = "agentsec-strict"
}

variable "model_armor_location" {
  description = "Region of the Model Armor template (templates are regional). Defaults to var.region. The Vertex AI floor-setting integration is only available in a subset of regions (europe-west1/2/3, asia-southeast1, asia-south1 per docs/sources.md); the SDK-driven screener works anywhere Model Armor is offered."
  type        = string
  default     = null
}

variable "model_armor_integrated_services" {
  description = "Services the project floor setting applies to. REST enum value for Vertex AI is AI_PLATFORM (gcloud spells it --add-integrated-services=VERTEX_AI)."
  type        = list(string)
  default     = ["AI_PLATFORM"]
}

variable "model_armor_floor_inspect_and_block" {
  description = "false = INSPECT_ONLY on the Vertex AI floor (visibility first); true = INSPECT_AND_BLOCK. Primer §11.3 discusses the trade-off."
  type        = bool
  default     = false
}

# ------------------------------------------------------------------------------------------
# Identity plumbing (CI, WIF)
# ------------------------------------------------------------------------------------------

variable "github_repo" {
  description = "GitHub repository (owner/name) allowed to impersonate the CI deployer through Workload Identity Federation. null disables the WIF pool/provider."
  type        = string
  default     = null
}

variable "ci_service_account_id" {
  description = "Account ID of the keyless CI deployer service account."
  type        = string
  default     = "agentsec-ci-deployer"
}

variable "mcp_service_account_id" {
  description = "Account ID of the fallback runtime identity for the MCP server (used until the service is switched to --identity-type=agent-identity)."
  type        = string
  default     = "agentsec-mcp-server"
}

# ------------------------------------------------------------------------------------------
# Observability
# ------------------------------------------------------------------------------------------

variable "audit_log_name" {
  description = "Cloud Logging log name the application's AuditLog writes to (AGENTSEC_AUDIT_LOG)."
  type        = string
  default     = "agentsec-audit"
}

variable "audit_dataset_id" {
  description = "BigQuery dataset that receives agent audit logs."
  type        = string
  default     = "agentsec_audit"
}

variable "bigquery_location" {
  description = "Location of the audit dataset (multi-region or region)."
  type        = string
  default     = "US"
}

variable "enable_data_access_audit_logs" {
  description = "Turn on DATA_READ/DATA_WRITE/ADMIN_READ audit logs for allServices. Required to see agent principals on data-plane calls; can be noisy/costly in busy projects."
  type        = bool
  default     = true
}

# ------------------------------------------------------------------------------------------
# Opt-in governance layers
# ------------------------------------------------------------------------------------------

variable "enable_agent_gateway" {
  description = "Create an egress (agent-to-anywhere) Agent Gateway with IAP + Model Armor authorization extensions and route the agent's outbound traffic through it."
  type        = bool
  default     = false
}

variable "agent_gateway_name" {
  description = "Name of the Agent Gateway."
  type        = string
  default     = "agentsec-egress-gateway"
}

variable "agent_gateway_iap_dry_run" {
  description = "true = IAP authorization extension in DRY_RUN (audit-only) mode; false = enforce."
  type        = bool
  default     = false
}

variable "enable_vpc_sc" {
  description = "Create a VPC Service Controls perimeter around the project (requires org-level Access Context Manager policy: var.access_policy_id)."
  type        = bool
  default     = false
}

variable "access_policy_id" {
  description = "Numeric Access Context Manager policy ID (organisation-level) used when enable_vpc_sc = true."
  type        = string
  default     = null
}

variable "vpc_sc_mcp_project_number" {
  description = "Project number hosting the MCP server for the egress rule. Defaults to var.project_number (same project)."
  type        = string
  default     = null
}

variable "enable_org_policy" {
  description = "Create Org Policy custom constraints (requires var.org_id and roles/orgpolicy.policyAdmin on the organisation)."
  type        = bool
  default     = false
}

variable "enable_pab" {
  description = "Create an organisation-level Principal Access Boundary policy that confines principals of this project to this project's resources (requires var.org_id)."
  type        = bool
  default     = false
}

variable "enable_agent_deny_policy" {
  description = "Create the project-level IAM deny policy for every agent in the project (never-list: object deletion, service-account key creation)."
  type        = bool
  default     = true
}
