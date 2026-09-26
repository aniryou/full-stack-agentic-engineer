# A bucket for model weights, readable by exactly one Kubernetes ServiceAccount through Workload
# Identity Federation (the principal:// form - no Google service account, no keys). The serving
# example mounts it with the Cloud Storage FUSE CSI driver.

data "google_project" "this" {
  project_id = var.project_id
}

locals {
  weights_principal = "principal://iam.googleapis.com/projects/${data.google_project.this.number}/locations/global/workloadIdentityPools/${var.project_id}.svc.id.goog/subject/ns/${var.weights_namespace}/sa/${var.weights_ksa}"
}

resource "google_storage_bucket" "weights" {
  count                       = var.create_weights_bucket ? 1 : 0
  name                        = "${var.project_id}-${var.cluster_name}-weights"
  location                    = var.weights_bucket_location
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = true # a lab bucket: `terraform destroy` removes the copied weights too

  soft_delete_policy {
    retention_duration_seconds = 0 # no soft-delete storage bill for re-downloadable weights
  }

  depends_on = [google_project_service.apis]
}

resource "google_storage_bucket_iam_member" "weights_reader" {
  count  = var.create_weights_bucket ? 1 : 0
  bucket = google_storage_bucket.weights[0].name
  role   = "roles/storage.objectViewer" # get + list: what GCS FUSE needs for a read-only mount
  member = local.weights_principal

  # The workload identity pool (<project>.svc.id.goog) exists once the cluster does.
  depends_on = [google_container_cluster.lab]
}
