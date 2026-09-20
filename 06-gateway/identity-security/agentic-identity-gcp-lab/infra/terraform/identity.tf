
# Identity plumbing that is *not* the agent's identity: the keyless CI deployer, the MCP
# server's fallback service account, and the artifacts they need. There are deliberately NO
# google_service_account_key resources anywhere in this configuration — long-lived keys are the
# thing Agent Identity, WIF and impersonation exist to replace (primer §3.4, §5).

# --- CI deployer -----------------------------------------------------------------------------

resource "google_service_account" "ci_deployer" {
  project      = var.project_id
  account_id   = var.ci_service_account_id
  display_name = "agentsec CI deployer (impersonated via WIF, no keys)"
  description  = "Deploys the MCP server image and the Agent Engine app from GitHub Actions."
  depends_on   = [google_project_service.apis]
}

# Minimal project roles for a deployer: build/push images, deploy Cloud Run, deploy Agent Engine.
resource "google_project_iam_member" "ci_deployer" {
  for_each = toset([
    "roles/artifactregistry.writer",
    "roles/cloudbuild.builds.editor",
    "roles/run.developer",
    "roles/aiplatform.user",
    "roles/serviceusage.serviceUsageConsumer",
    "roles/logging.viewer",
  ])

  project = var.project_id
  role    = each.value
  member  = google_service_account.ci_deployer.member
}

# Cloud Run deployments that run as the fallback SA need actAs on it (not on any other SA).
resource "google_service_account_iam_member" "ci_acts_as_mcp_sa" {
  service_account_id = google_service_account.mcp_server.name
  role               = "roles/iam.serviceAccountUser"
  member             = google_service_account.ci_deployer.member
}

# --- Workload Identity Federation for GitHub Actions ---------------------------------------
# WIF = keyless CI: GitHub mints an OIDC token for the workflow run, STS exchanges it for a
# short-lived Google credential, which may then impersonate ci_deployer. The
# attribute_condition pins the exchange to ONE repository, so a token from any other repo on
# the same issuer is rejected before IAM is even consulted (primer §3.4, "WIF").

resource "google_iam_workload_identity_pool" "github" {
  count = var.github_repo == null ? 0 : 1

  project                   = var.project_id
  workload_identity_pool_id = "agentsec-github"
  display_name              = "agentsec GitHub Actions"
  description               = "OIDC federation for the lab's CI workflows"
  depends_on                = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  count = var.github_repo == null ? 0 : 1

  project                            = var.project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.github[0].workload_identity_pool_id
  workload_identity_pool_provider_id = "github-oidc"
  display_name                       = "GitHub OIDC"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.actor"      = "assertion.actor"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }

  # Only workflows from this repository may federate. Tighten further with
  # `&& assertion.ref == "refs/heads/main"` for production.
  attribute_condition = "assertion.repository == \"${var.github_repo}\""

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# Let identities from that repository impersonate the deployer (no key download involved).
resource "google_service_account_iam_member" "github_impersonates_ci" {
  count = var.github_repo == null ? 0 : 1

  service_account_id = google_service_account.ci_deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github[0].name}/attribute.repository/${var.github_repo}"
}

# --- MCP server fallback identity --------------------------------------------------------------
# Used by the Cloud Run service until deploy_mcp_cloud_run.sh switches it to
# --identity-type=agent-identity. Migration note (docs/sources.md): the agent identity is a NEW
# principal that inherits nothing from this SA — grant it explicitly (see iam.tf / grant_agent_iam.sh).

resource "google_service_account" "mcp_server" {
  project      = var.project_id
  account_id   = var.mcp_service_account_id
  display_name = "agentsec MCP server (fallback runtime identity)"
  depends_on   = [google_project_service.apis]
}

resource "google_project_iam_member" "mcp_server_logs" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = google_service_account.mcp_server.member
}

# --- Artifacts: image repo + Agent Engine staging bucket -------------------------------------

resource "google_artifact_registry_repository" "images" {
  project       = var.project_id
  location      = var.region
  repository_id = "agentsec"
  format        = "DOCKER"
  description   = "MCP server images for the agentsec lab"
  labels        = local.labels
  depends_on    = [google_project_service.apis]
}

# Staging bucket for the Python-SDK deployment path (deploy_agent_engine.py uploads the app here).
resource "google_storage_bucket" "agent_staging" {
  project                     = var.project_id
  name                        = "${var.project_id}-agentsec-staging"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = true
  labels                      = local.labels
  depends_on                  = [google_project_service.apis]
}

resource "google_storage_bucket_iam_member" "ci_staging_writer" {
  bucket = google_storage_bucket.agent_staging.name
  role   = "roles/storage.objectAdmin"
  member = google_service_account.ci_deployer.member
}
