
# The tickets MCP server (src/agentsec/mcp/server.py) as a Cloud Run service: an OAuth 2.1
# resource server that validates audience, maps scopes to tools and never forwards inbound
# tokens (primer §7.1). Network posture: internal ingress + IAM invoker, i.e. reachable only
# by principals we name — the agent — and only over Agent Gateway / PSC when the gateway is on.
#
# NOTE on Agent Platform flags: `--functional-type=mcp-server --identity-type=agent-identity`
# are beta gcloud flags with no Terraform equivalent in google 8.1.0. Terraform creates the
# service with the fallback service account; infra/scripts/deploy_mcp_cloud_run.sh then runs
#   gcloud beta run deploy <SERVICE> --image=... --functional-type=mcp-server \
#     --identity-type=agent-identity --no-allow-unauthenticated --ingress=internal --region=...
# which also auto-registers the service in Agent Registry under /mcpServers. The lifecycle
# block below keeps Terraform from reverting what that script sets.

resource "google_cloud_run_v2_service" "mcp" {
  project             = var.project_id
  name                = var.mcp_service_name
  location            = var.region
  description         = "agentsec tickets MCP server (OAuth 2.1 resource server)"
  ingress             = var.mcp_ingress
  deletion_protection = false
  labels              = local.labels

  template {
    service_account                  = google_service_account.mcp_server.email
    execution_environment            = "EXECUTION_ENVIRONMENT_GEN2"
    timeout                          = "300s"
    max_instance_request_concurrency = 80
    labels                           = local.labels

    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }

    containers {
      name  = "mcp"
      image = local.mcp_server_image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
        cpu_idle          = true
        startup_cpu_boost = true
      }

      startup_probe {
        initial_delay_seconds = 2
        period_seconds        = 5
        failure_threshold     = 6
        timeout_seconds       = 3
        http_get {
          path = "/healthz"
          port = 8080
        }
      }

      env {
        name  = "AGENTSEC_PROFILE"
        value = "gcp"
      }
      env {
        name  = "AGENTSEC_PROJECT_ID"
        value = var.project_id
      }
      env {
        name  = "AGENTSEC_PROJECT_NUMBER"
        value = var.project_number
      }
      env {
        name  = "AGENTSEC_LOCATION"
        value = var.region
      }
      # The server's canonical URI = its RFC 8707 audience. Cloud Run's deterministic URL lets us
      # set it before the service exists (the computed `uri` cannot reference itself).
      env {
        name  = "AGENTSEC_MCP_URL"
        value = "https://${var.mcp_service_name}-${var.project_number}.${var.region}.run.app/mcp"
      }
      env {
        name  = "AGENTSEC_STS_ISSUER"
        value = var.mcp_sts_issuer
      }
      env {
        name  = "AGENTSEC_AUDIT_LOG"
        value = var.audit_log_name
      }
    }
  }

  lifecycle {
    ignore_changes = [
      template[0].containers[0].image, # CI / deploy_mcp_cloud_run.sh roll images
      template[0].service_account,     # replaced by Agent Identity after the gcloud switch
      template[0].annotations,
      annotations,
      client,
      client_version,
      launch_stage, # becomes BETA once beta (Agent Platform) features are in use
    ]
  }

  depends_on = [google_project_service.apis, google_artifact_registry_repository.images]
}

# Who may invoke the MCP server: the agent principal, nobody else (no allUsers, no project-wide
# run.invoker). The Cloud Run service is a policy enforcement point of its own (primer §4.1).
resource "google_cloud_run_v2_service_iam_member" "agent_invoker" {
  count = local.agent_identity_known ? 1 : 0

  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.mcp.name
  role     = "roles/run.invoker"
  member   = local.agent_principal
}
