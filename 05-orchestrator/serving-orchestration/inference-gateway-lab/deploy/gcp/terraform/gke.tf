# Zonal GKE Standard cluster with the Gateway API enabled and Managed Service for Prometheus on.
# Zonal keeps the control plane in one zone (cheapest). # VERIFY: GKE's management-fee credit covers one zonal cluster per billing account.
resource "google_container_cluster" "this" {
  name     = var.name
  location = var.zone

  network         = google_compute_network.vpc.id
  subnetwork      = google_compute_subnetwork.nodes.id
  networking_mode = "VPC_NATIVE"

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Node pools are managed separately below.
  remove_default_node_pool = true
  initial_node_count       = 1
  deletion_protection      = false

  release_channel {
    channel = var.release_channel
  }

  # Installs the Gateway API CRDs and GKE's GatewayClasses (gke-l7-regional-external-managed, gke-l7-rilb, ...).
  # VERIFY: on GKE >= 1.34.0-gke.1626000 the InferencePool v1 CRD is managed by GKE as well.
  gateway_api_config {
    channel = "CHANNEL_STANDARD"
  }

  monitoring_config {
    enable_components = var.monitoring_components
    managed_prometheus {
      enabled = true
    }
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  resource_labels = var.labels

  depends_on = [google_project_service.apis, google_compute_subnetwork.proxy_only]
}

# CPU pool: EPP, custom-metrics adapter, system pods. One small node.
resource "google_container_node_pool" "system" {
  name       = "system"
  cluster    = google_container_cluster.this.id
  location   = var.zone
  node_count = 1

  node_config {
    machine_type    = var.system_machine_type
    disk_type       = "pd-balanced"
    disk_size_gb    = 50
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    labels          = { pool = "system" }
    resource_labels = var.labels
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# GPU pool: one L4 per node, Spot, autoscaled 0 -> gpu_max_nodes by the cluster autoscaler when
# vLLM pods are pending. GKE installs the default NVIDIA driver and taints the nodes
# (nvidia.com/gpu=present:NoSchedule), so only pods that tolerate it land here.
resource "google_container_node_pool" "gpu" {
  name               = "l4-spot"
  cluster            = google_container_cluster.this.id
  location           = var.zone
  initial_node_count = 0

  autoscaling {
    min_node_count = 0
    max_node_count = var.gpu_max_nodes
  }

  node_config {
    machine_type = var.gpu_machine_type
    spot         = var.gpu_spot
    disk_type    = "pd-balanced"
    disk_size_gb = 100

    guest_accelerator {
      type  = var.gpu_type
      count = var.gpus_per_node

      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    # Image streaming: start the multi-GB vLLM image before it is fully pulled. # VERIFY availability for your node image.
    gcfs_config {
      enabled = true
    }

    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    labels          = { pool = "gpu" }
    resource_labels = var.labels
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}
