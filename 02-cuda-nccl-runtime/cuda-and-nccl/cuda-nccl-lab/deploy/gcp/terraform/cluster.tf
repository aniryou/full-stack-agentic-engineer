resource "google_container_cluster" "lab" {
  name     = var.cluster_name
  location = var.zone # zonal: one control plane, the cheapest shape (VERIFY the GKE free-tier credit)

  network         = google_compute_network.lab.id
  subnetwork      = google_compute_subnetwork.lab.id
  networking_mode = "VPC_NATIVE"

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Node pools are managed below, one resource per concern.
  remove_default_node_pool = true
  initial_node_count       = 1
  deletion_protection      = false # a lab: `terraform destroy` must work

  release_channel {
    channel = var.release_channel
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # DCGM: GKE runs the exporter on GPU nodes and ships DCGM_FI_* metrics to Cloud Monitoring via
  # Managed Prometheus (query them with PromQL; alert with deploy/gke/06-dcgm-alert-rules.yaml).
  monitoring_config {
    enable_components = var.enable_dcgm ? ["SYSTEM_COMPONENTS", "DCGM"] : ["SYSTEM_COMPONENTS"]

    managed_prometheus {
      enabled = true
    }
  }

  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }

  resource_labels = var.labels

  depends_on = [google_project_service.apis]
}
