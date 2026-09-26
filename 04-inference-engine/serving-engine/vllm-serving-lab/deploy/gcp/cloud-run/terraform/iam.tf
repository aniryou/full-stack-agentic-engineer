# Identity: the service runs as its own service account with exactly two optional grants
# (read the HF token, read the weights bucket), and only named principals may invoke it.

resource "google_service_account" "vllm" {
  project      = var.project_id
  account_id   = substr("${var.service_name}-run", 0, 30)
  display_name = "vLLM on Cloud Run (${var.service_name})"
}

resource "google_secret_manager_secret_iam_member" "hf_token" {
  count = var.hf_token_secret_id == null ? 0 : 1

  project   = var.project_id
  secret_id = var.hf_token_secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.vllm.email}"
}

resource "google_storage_bucket_iam_member" "weights_reader" {
  count = var.model_source == "gcs" ? 1 : 0

  bucket = var.create_weights_bucket ? google_storage_bucket.weights[0].name : var.weights_bucket
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.vllm.email}"
}

resource "google_cloud_run_v2_service_iam_member" "invokers" {
  for_each = toset(var.invoker_members)

  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.vllm.name
  role     = "roles/run.invoker"
  member   = each.value
}
