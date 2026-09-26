# Zonal GKE Standard: one control plane in one zone (the cheapest shape; GKE's free-tier credit
# covers the management fee of one zonal cluster per billing account - verify). The default node
# pool is removed; node_pools.tf owns every node.

resource "google_container_cluster" "lab" {
  name     = var.cluster_name
  location = var.zone

  network         = google_compute_network.lab.id
  subnetwork      = google_compute_subnetwork.lab.id
  networking_mode = "VPC_NATIVE"

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  remove_default_node_pool = true
  initial_node_count       = 1
  deletion_protection      = var.deletion_protection

  release_channel {
    channel = var.release_channel
  }

  # Workload Identity: pods authenticate to Google APIs as their Kubernetes ServiceAccount
  # (storage.tf grants the serving KSA read access to the weights bucket - no keys anywhere).
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  addons_config {
    gcs_fuse_csi_driver_config {
      enabled = var.enable_gcs_fuse
    }
  }

  # Managed Prometheus: scrape DCGM / vLLM / Kueue metrics with PodMonitoring objects.
  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS"]
    managed_prometheus {
      enabled = true
    }
  }

  # Image streaming for every node pool (each pool also sets it explicitly).
  node_pool_defaults {
    node_config_defaults {
      gcfs_config {
        enabled = var.enable_image_streaming
      }
    }
  }

  depends_on = [google_project_service.apis]
}
