output "cluster_name" {
  description = "GKE cluster name."
  value       = google_container_cluster.lab.name
}

output "location" {
  description = "Zone of the cluster."
  value       = google_container_cluster.lab.location
}

output "get_credentials" {
  description = "Point kubectl at the cluster."
  value       = "gcloud container clusters get-credentials ${google_container_cluster.lab.name} --location ${google_container_cluster.lab.location} --project ${var.project_id}"
}

output "gpu_pools" {
  description = "GPU node pools (cloud.google.com/gke-nodepool label values)."
  value       = concat([google_container_node_pool.gpu_spot.name], [for p in google_container_node_pool.gpu_flex : p.name])
}

output "weights_bucket" {
  description = "Bucket for model weights (null when create_weights_bucket = false)."
  value       = var.create_weights_bucket ? google_storage_bucket.weights[0].name : null
}

output "weights_reader_principal" {
  description = "The Workload Identity principal that may read the weights bucket."
  value       = local.weights_principal
}

output "next_steps" {
  description = "What to run after apply."
  value       = <<-EOT
    gcloud container clusters get-credentials ${google_container_cluster.lab.name} --location ${google_container_cluster.lab.location} --project ${var.project_id}
    deploy/gke/apply-examples.sh smoke          # scale the L4 Spot pool from zero, run nvidia-smi
    deploy/gke/install-addons.sh                # JobSet + Kueue + the GKE queues (DWS needs enable_flex_start_pool)
    deploy/gke/apply-examples.sh dws
    WEIGHTS_BUCKET=<weights_bucket> deploy/gke/apply-examples.sh serving
    terraform destroy                           # when you are done: nothing keeps billing
  EOT
}
