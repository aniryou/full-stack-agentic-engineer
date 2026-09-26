output "cluster_name" {
  description = "GKE cluster name."
  value       = google_container_cluster.this.name
}

output "location" {
  description = "Cluster zone."
  value       = google_container_cluster.this.location
}

output "get_credentials" {
  description = "Command that points kubectl at the cluster."
  value       = "gcloud container clusters get-credentials ${google_container_cluster.this.name} --zone ${var.zone} --project ${var.project_id}"
}

output "proxy_only_subnet" {
  description = "Proxy-only subnet used by the regional Gateway's managed proxies."
  value       = google_compute_subnetwork.proxy_only.ip_cidr_range
}

output "next_steps" {
  description = "What to run after apply."
  value       = "PROJECT_ID=${var.project_id} ZONE=${var.zone} CLUSTER=${google_container_cluster.this.name} ../../gke/install.sh"
}
