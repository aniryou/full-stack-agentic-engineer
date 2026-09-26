# Service enablement and the optional weights bucket.

resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",           # the service
    "iam.googleapis.com",           # its service account
    "secretmanager.googleapis.com", # the optional HF token
    "storage.googleapis.com",       # the optional weights bucket
  ])

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# Weights in Cloud Storage, mounted read-only at /models: cold starts stop depending on the
# Hugging Face Hub and do not fill the instance's in-memory filesystem with a download.
#   gcloud storage cp -r ./Qwen2.5-1.5B-Instruct gs://<bucket>/Qwen2.5-1.5B-Instruct
resource "google_storage_bucket" "weights" {
  count = var.create_weights_bucket ? 1 : 0

  project                     = var.project_id
  name                        = var.weights_bucket
  location                    = var.region # same region as the service: no cross-region egress
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = true # lab bucket: `terraform destroy` removes the weights too
  labels                      = var.labels

  depends_on = [google_project_service.apis]
}
