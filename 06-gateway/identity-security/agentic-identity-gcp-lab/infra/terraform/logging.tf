
# Audit and observability (primer §9): every action must be attributable to *both* identities
# (the agent, and the user it acted for). Two sources feed one BigQuery dataset:
#   1. platform Cloud Audit Logs where the caller is an agent principal
#      (protoPayload.authenticationInfo.principalSubject contains the agents.* trust domain);
#   2. the application's own structured audit events (AuditLog -> CloudLoggingSink), written to
#      the `agentsec-audit` log with the dual identity in jsonPayload.

# Data Access audit logs are off by default for most services; without them the agent's reads
# (Secret Manager access, Vertex calls) are invisible. Cost note: allServices DATA_READ can be
# voluminous in busy projects — scope to specific services in production, or add
# exempted_members for chatty service accounts.
resource "google_project_iam_audit_config" "all_services" {
  count = var.enable_data_access_audit_logs ? 1 : 0

  project = var.project_id
  service = "allServices"

  audit_log_config {
    log_type = "ADMIN_READ"
  }
  audit_log_config {
    log_type = "DATA_READ"
  }
  audit_log_config {
    log_type = "DATA_WRITE"
  }
}

resource "google_bigquery_dataset" "audit" {
  project                     = var.project_id
  dataset_id                  = var.audit_dataset_id
  friendly_name               = "agentsec agent audit trail"
  description                 = "Cloud Audit Logs for agent principals + agentsec application audit events"
  location                    = var.bigquery_location
  delete_contents_on_destroy  = true # lab: destroy cleans up; remove for production retention
  default_table_expiration_ms = 90 * 24 * 60 * 60 * 1000
  labels                      = local.labels

  depends_on = [google_project_service.apis]
}

resource "google_logging_project_sink" "agent_audit" {
  project     = var.project_id
  name        = "agentsec-agent-audit"
  description = "Route agent-principal audit logs and agentsec audit events to BigQuery"
  destination = "bigquery.googleapis.com/projects/${var.project_id}/datasets/${google_bigquery_dataset.audit.dataset_id}"

  # Either the app's own audit log, or any audit-log entry whose caller is an agent identity.
  filter = <<-EOT
    logName="projects/${var.project_id}/logs/${var.audit_log_name}"
    OR (
      logName:"cloudaudit.googleapis.com"
      AND protoPayload.authenticationInfo.principalSubject:"agents.global"
    )
  EOT

  # A dedicated writer identity per sink (not the shared legacy cloud-logs account).
  unique_writer_identity = true

  bigquery_options {
    use_partitioned_tables = true # one partitioned table per log instead of dated tables
  }
}

# The sink's writer identity needs to be able to create/append tables in the dataset.
resource "google_bigquery_dataset_iam_member" "sink_writer" {
  project    = var.project_id
  dataset_id = google_bigquery_dataset.audit.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = google_logging_project_sink.agent_audit.writer_identity
}
