# Shared names and labels, and the APIs the lab needs.

resource "random_id" "suffix" {
  byte_length = 3
}

locals {
  name   = "${var.name}-${random_id.suffix.hex}"
  labels = merge(var.labels, { "managed-by" = "terraform" })
  apis   = ["compute.googleapis.com", "storage.googleapis.com", "iam.googleapis.com", "logging.googleapis.com"]
}

resource "google_project_service" "apis" {
  for_each = var.enable_apis ? toset(local.apis) : toset([])

  service            = each.value
  disable_on_destroy = false
}
