# Zonal GKE Standard with private nodes, Dataplane V2 (which enforces NetworkPolicy), Workload Identity
# and Managed Prometheus. The default node pool is removed; node_pools.tf owns every node.

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

  # Dataplane V2 (Cilium-based): NetworkPolicy is enforced without the Calico add-on.
  datapath_provider = "ADVANCED_DATAPATH"

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = var.master_cidr
  }

  dynamic "master_authorized_networks_config" {
    for_each = length(var.authorized_networks) > 0 ? [1] : []
    content {
      dynamic "cidr_blocks" {
        for_each = var.authorized_networks
        content {
          cidr_block   = cidr_blocks.value
          display_name = "lab operator"
        }
      }
    }
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  addons_config {
    # VERIFY: the Agent Sandbox add-on field exists in provider 8.4.0; its availability per release channel is not checked here.
    agent_sandbox_config {
      enabled = var.enable_agent_sandbox_addon
    }
  }

  # Managed Prometheus: execution latency and kill reasons as metrics (PodMonitoring objects).
  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS"]
    managed_prometheus {
      enabled = true
    }
  }

  depends_on = [google_project_service.apis]
}
