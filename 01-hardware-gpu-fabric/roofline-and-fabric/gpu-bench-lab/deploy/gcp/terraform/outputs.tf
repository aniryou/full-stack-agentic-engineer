output "instance_name" {
  description = "The benchmark VM."
  value       = google_compute_instance.bench.name
}

output "results_uri" {
  description = "Where the VM uploads its report (one folder per run)."
  value       = "gs://${google_storage_bucket.results.name}/results/"
}

output "watch_progress" {
  description = "Follow the startup script (it prints every step with a [gpubench] prefix)."
  value       = "gcloud compute instances get-serial-port-output ${google_compute_instance.bench.name} --zone ${var.zone} --project ${var.project_id} | grep gpubench"
}

output "fetch_results" {
  description = "Copy the reports to your machine."
  value       = "gcloud storage cp -r gs://${google_storage_bucket.results.name}/results ./results-gcp"
}

output "ssh" {
  description = "SSH through IAP (no public SSH port is open)."
  value       = "gcloud compute ssh ${google_compute_instance.bench.name} --zone ${var.zone} --project ${var.project_id} --tunnel-through-iap"
}

output "cleanup" {
  description = "Delete everything this configuration created (including the bucket and its reports)."
  value       = "terraform destroy -var project_id=${var.project_id}"
}
