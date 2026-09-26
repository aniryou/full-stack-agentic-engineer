output "cluster_name" {
  value = google_container_cluster.lab.name
}

output "zone" {
  value = var.zone
}

output "get_credentials" {
  description = "Point kubectl at the cluster."
  value       = "gcloud container clusters get-credentials ${google_container_cluster.lab.name} --zone ${var.zone} --project ${var.project_id}"
}

output "gpu_pools" {
  description = "GPU node pools and what they offer (all scale from zero)."
  value = {
    for name, pool in google_container_node_pool.gpu : name => {
      machine_type = pool.node_config[0].machine_type
      accelerator  = "${pool.node_config[0].guest_accelerator[0].count} x ${pool.node_config[0].guest_accelerator[0].type}"
      spot         = pool.node_config[0].spot
    }
  }
}

output "next_steps" {
  value = <<-EOT
    $(terraform output -raw get_credentials)
    cd ../../gke && ./run.sh status && ./run.sh smoke && ./run.sh nccl
    Cloud Monitoring -> Metrics explorer -> PromQL: DCGM_FI_PROF_SM_ACTIVE, DCGM_FI_DEV_GPU_UTIL
    When done: kubectl delete namespace gpu-lab && terraform destroy
  EOT
}
