# APIs the lab needs. disable_on_destroy = false so `terraform destroy` never turns off an API
# other workloads in the project may use.
resource "google_project_service" "apis" {
  for_each = var.enable_apis ? toset([
    "compute.googleapis.com",
    "container.googleapis.com",
    "monitoring.googleapis.com",
  ]) : toset([])

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}
