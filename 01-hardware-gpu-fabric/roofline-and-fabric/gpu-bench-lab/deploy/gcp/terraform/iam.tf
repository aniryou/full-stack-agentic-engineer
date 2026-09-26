# The VM runs as its own service account that can write objects to one bucket and write logs —
# nothing else in the project.

resource "google_service_account" "bench" {
  account_id   = "${local.name}-vm"
  display_name = "gpubench benchmark VM"
  depends_on   = [google_project_service.apis]
}

resource "google_storage_bucket_iam_member" "results_writer" {
  bucket = google_storage_bucket.results.name
  role   = "roles/storage.objectUser" # read/write objects in this bucket only (gcloud storage cp)
  member = "serviceAccount:${google_service_account.bench.email}"
}

resource "google_project_iam_member" "log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter" # the startup script's output reaches Cloud Logging
  member  = "serviceAccount:${google_service_account.bench.email}"
}
