
# The ONE demo secret the agent may read (primer §5: secrets are fetched at call time by the
# runtime, never pasted into prompts or session state). It is injected into the Agent Engine
# container as AGENTSEC_SECRET_DEMO (EnvSecretStore's AGENTSEC_SECRET_<NAME> convention).

resource "google_secret_manager_secret" "demo" {
  project   = var.project_id
  secret_id = "agentsec-demo-secret"
  labels    = local.labels

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

# Write-only payload: the value is sent to Secret Manager but not persisted in Terraform state.
# Bump var.demo_secret_version to rotate.
resource "google_secret_manager_secret_version" "demo" {
  secret                 = google_secret_manager_secret.demo.id
  secret_data_wo         = var.demo_secret_value
  secret_data_wo_version = var.demo_secret_version
}

# Accessor binding for the agent principal ONLY — on this secret, not at project level.
# Blast radius of a compromised agent = exactly this one secret (primer §4.5 "envelope").
resource "google_secret_manager_secret_iam_member" "agent_accessor" {
  count = local.agent_identity_known ? 1 : 0

  project   = var.project_id
  secret_id = google_secret_manager_secret.demo.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = local.agent_principal
}

# When Terraform deploys the reasoning engine, the platform resolves `secret_env` at deploy
# time. The provider docs say the Reasoning Engine *service agent* needs secretAccessor for that.
# VERIFY: whether this is still required when identity_type = AGENT_IDENTITY (the agent principal
# above may be sufficient); the service-agent email pattern is the documented one for Agent Engine.
resource "google_secret_manager_secret_iam_member" "reasoning_engine_service_agent" {
  count = var.deploy_agent_with_terraform ? 1 : 0

  project   = var.project_id
  secret_id = google_secret_manager_secret.demo.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:service-${var.project_number}@gcp-sa-aiplatform-re.iam.gserviceaccount.com"
}
