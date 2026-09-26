# Where the VM puts its report: a private bucket that empties itself after a while.

resource "google_storage_bucket" "results" {
  name                        = "${var.project_id}-${local.name}"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = true # `terraform destroy` removes the reports too: download them first
  labels                      = local.labels

  lifecycle_rule {
    condition {
      age = var.results_retention_days
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.apis]
}
