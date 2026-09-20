
# Shared derived values. The identity strings below are the heart of the lab (primer §3.3):
# an agent is a first-class IAM principal whose name is *derived from the resource it runs as*,
# so there is no key to leak and nothing to impersonate — the platform issues the certificate.

locals {
  labels = var.labels

  # Trust domain: organisation-scoped when the project belongs to an org, otherwise
  # project-scoped (docs/sources.md, "Agent Identity").
  trust_domain = var.org_id != null ? "agents.global.org-${var.org_id}.system.id.goog" : "agents.global.project-${var.project_number}.system.id.goog"

  # Which reasoning engine ID do we bind policies to?
  #  * deploy_agent_with_terraform = true  -> the ID Terraform just created (numeric, last path segment)
  #  * otherwise                           -> var.agent_engine_id (from infra/scripts/deploy_agent_engine.py)
  reasoning_engine_name_parts = var.deploy_agent_with_terraform ? split("/", google_vertex_ai_reasoning_engine.agent[0].name) : []
  agent_engine_id             = var.deploy_agent_with_terraform ? element(local.reasoning_engine_name_parts, length(local.reasoning_engine_name_parts) - 1) : var.agent_engine_id

  # `count` must be decidable at plan time, so gate on variables only (the ID itself may be
  # unknown until apply when Terraform deploys the engine).
  agent_identity_known = var.deploy_agent_with_terraform || var.agent_engine_id != null

  # The agent's IAM principal:
  #   principal://TRUST_DOMAIN/resources/aiplatform/projects/PROJECT_NUMBER/locations/REGION/reasoningEngines/ID
  agent_principal = local.agent_identity_known ? "principal://${local.trust_domain}/resources/aiplatform/projects/${var.project_number}/locations/${var.region}/reasoningEngines/${local.agent_engine_id}" : null

  # Every agent hosted by Agent Engine in this project (fleet-level policies, primer §4.5).
  all_project_agents = "principalSet://${local.trust_domain}/attribute.platformContainer/aiplatform/projects/${var.project_number}"

  # The MCP server's *own* agent identity once deploy_mcp_cloud_run.sh switches the Cloud Run
  # service to --identity-type=agent-identity (Cloud Run docs: resources/run/.../services/NAME).
  mcp_server_principal = "principal://${local.trust_domain}/resources/run/projects/${var.project_number}/locations/${var.region}/services/${var.mcp_service_name}"

  # Derived defaults
  auth_provider_location = coalesce(var.auth_provider_location, var.region)
  model_armor_location   = coalesce(var.model_armor_location, var.region)
  mcp_server_image       = coalesce(var.mcp_server_image, "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}/mcp-tickets:latest")
  mcp_server_url         = "${google_cloud_run_v2_service.mcp.uri}/mcp"
  vpc_sc_mcp_project     = coalesce(var.vpc_sc_mcp_project_number, var.project_number)
}
