
output "trust_domain" {
  description = "Agent Identity trust domain for this project/org."
  value       = local.trust_domain
}

output "agent_principal" {
  description = "IAM principal of the deployed support agent (principal://...). null until an engine ID is known."
  value       = local.agent_principal
}

output "agent_spiffe_id" {
  description = "SPIFFE ID of the agent (same path as the principal, spiffe:// scheme)."
  value       = local.agent_identity_known ? replace(local.agent_principal, "principal://", "spiffe://") : null
}

output "all_project_agents_principal_set" {
  description = "principalSet selecting every Agent Engine agent in this project (fleet policies)."
  value       = local.all_project_agents
}

output "reasoning_engine_name" {
  description = "Full resource name of the reasoning engine when deployed by Terraform."
  value       = var.deploy_agent_with_terraform ? google_vertex_ai_reasoning_engine.agent[0].name : null
}

output "agent_engine_id" {
  description = "Reasoning engine ID used for IAM bindings."
  value       = local.agent_engine_id
}

output "mcp_server_url" {
  description = "Canonical MCP endpoint (RFC 8707 resource indicator / token audience)."
  value       = local.mcp_server_url
}

output "mcp_server_principal" {
  description = "Agent Identity principal of the MCP server once switched to --identity-type=agent-identity."
  value       = local.mcp_server_principal
}

output "mcp_server_registry_id" {
  description = "Agent Registry MCP server ID (for gcloud iap web ... --mcp-server=)."
  value       = local.mcp_server_registry_id
}

output "mcp_server_image" {
  description = "Image the Cloud Run service was created with."
  value       = local.mcp_server_image
}

output "auth_provider_name" {
  description = "Auth Manager provider resource name (AGENTSEC_AUTH_PROVIDER)."
  value       = google_agent_identity_auth_provider.crm.name
}

output "auth_provider_redirect_url" {
  description = "Deterministic OAuth callback URL to register at the CRM identity provider."
  value       = google_agent_identity_auth_provider.crm.auth_provider_type_params[0].three_legged_oauth[0].redirect_url
}

output "agent_gateway_name" {
  description = "Agent Gateway resource name (null when var.enable_agent_gateway = false)."
  value       = var.enable_agent_gateway ? google_network_services_agent_gateway.egress[0].id : null
}

output "agent_gateway_mtls_endpoint" {
  description = "mTLS endpoint of the gateway (null when disabled)."
  value       = var.enable_agent_gateway ? google_network_services_agent_gateway.egress[0].agent_gateway_card[0].mtls_endpoint : null
}

output "model_armor_template_name" {
  description = "Model Armor template resource name (AGENTSEC_MODEL_ARMOR_TEMPLATE)."
  value       = google_model_armor_template.strict.name
}

output "demo_secret_id" {
  description = "Secret Manager secret the agent may read."
  value       = google_secret_manager_secret.demo.secret_id
}

output "ci_service_account_email" {
  description = "Keyless CI deployer (impersonate via WIF)."
  value       = google_service_account.ci_deployer.email
}

output "workload_identity_provider" {
  description = "WIF provider resource name for google-github-actions/auth (null when github_repo unset)."
  value       = var.github_repo == null ? null : google_iam_workload_identity_pool_provider.github[0].name
}

output "agent_staging_bucket" {
  description = "gs:// bucket used by deploy_agent_engine.py."
  value       = "gs://${google_storage_bucket.agent_staging.name}"
}

output "audit_dataset" {
  description = "BigQuery dataset receiving agent audit logs."
  value       = "${var.project_id}.${google_bigquery_dataset.audit.dataset_id}"
}

output "agent_env" {
  description = "Environment the agent runtime is configured with (handy for deploy_agent_engine.py --env-from-terraform)."
  value       = local.agent_env
}
