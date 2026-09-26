# A node service account with only what nodes need (logs, metrics, image pulls) - instead of the
# default compute service account. Pods never see it: GKE_METADATA mode on the node pools serves
# pods their Kubernetes ServiceAccount's Workload Identity, and the sandbox's KSA has no IAM binding.

resource "google_service_account" "nodes" {
  account_id   = "${var.cluster_name}-nodes"
  display_name = "sandbox-lab GKE nodes"
}

locals {
  node_roles = [
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer",
    "roles/artifactregistry.reader",
  ]
}

resource "google_project_iam_member" "nodes" {
  for_each = toset(local.node_roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.nodes.email}"
}
