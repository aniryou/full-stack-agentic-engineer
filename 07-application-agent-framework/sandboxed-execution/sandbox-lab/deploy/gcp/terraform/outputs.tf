output "project_id" {
  description = "Project of the cluster (read by deploy/gke/apply.sh)."
  value       = var.project_id
}

output "zone" {
  description = "Zone of the cluster."
  value       = var.zone
}

output "cluster_name" {
  description = "GKE cluster name."
  value       = google_container_cluster.lab.name
}

output "get_credentials" {
  description = "Command that points kubectl at the cluster."
  value       = "gcloud container clusters get-credentials ${google_container_cluster.lab.name} --zone ${var.zone} --project ${var.project_id}"
}

output "repository_url" {
  description = "Artifact Registry path that replaces LOCATION-docker.pkg.dev/PROJECT_ID/sandbox in deploy/gke/*.yaml."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.sandbox.repository_id}"
}

output "services_cidr" {
  description = "Service ClusterIP range; the egress proxy's fixed ClusterIP must be inside it."
  value       = var.services_cidr
}

output "sandbox_pool_taint" {
  description = "The taint GKE puts on GKE Sandbox nodes (verify with kubectl describe node)."
  value       = local.gke_sandbox_taint
}

output "internet_egress" {
  description = "Whether anything in the cluster can reach the internet."
  value       = var.enable_nat ? "Cloud NAT on: only NetworkPolicy keeps sandboxes off the internet (gVisor's default network stack allows egress)" : "no route to the internet (no NAT, private nodes)"
}
