# Services the lab needs. disable_on_destroy = false: turning an API off on `terraform destroy`
# could break other things in the same project.

locals {
  apis = [
    "compute.googleapis.com",
    "container.googleapis.com",
    "storage.googleapis.com",
    "monitoring.googleapis.com", # managed Prometheus writes here
  ]
}

resource "google_project_service" "apis" {
  for_each           = toset(local.apis)
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}
