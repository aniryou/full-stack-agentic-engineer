# Nodes run as a dedicated, minimal service account instead of the Compute Engine default one.

resource "google_service_account" "nodes" {
  account_id   = "${var.cluster_name}-nodes"
  display_name = "GKE nodes for ${var.cluster_name}"
  depends_on   = [google_project_service.apis]
}

resource "google_project_iam_member" "nodes" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer",
    "roles/autoscaling.metricsWriter",
    "roles/artifactregistry.reader",
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.nodes.email}"
}
