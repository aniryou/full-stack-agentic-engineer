# Three pools, one per way of getting capacity:
#   system   - always-on CPU node for controllers (kube-system, Kueue, JobSet)
#   l4-spot  - L4 GPUs on Spot, autoscaling 0..N: you pay only while a GPU pod runs
#   l4-flex  - (optional) DWS flex-start with queued provisioning: Kueue files a ProvisioningRequest and
#              DWS creates every node of the gang at once, for at most 7 days
# The GPU pools carry the nvidia.com/gpu=present:NoSchedule taint explicitly (GKE applies the same
# taint to GPU nodes - verify), so only pods that tolerate it (or request GPUs through GKE's
# ExtendedResourceToleration admission) land there.

locals {
  gpu_taint = {
    key    = "nvidia.com/gpu"
    value  = "present"
    effect = "NO_SCHEDULE"
  }
  oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]
}

resource "google_container_node_pool" "system" {
  name       = "system"
  cluster    = google_container_cluster.lab.id
  location   = var.zone
  node_count = var.system_node_count

  node_config {
    machine_type    = var.system_machine_type
    disk_size_gb    = 50
    disk_type       = "pd-balanced"
    oauth_scopes    = local.oauth_scopes
    resource_labels = var.labels

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    gcfs_config {
      enabled = var.enable_image_streaming
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

resource "google_container_node_pool" "gpu_spot" {
  name               = "l4-spot"
  cluster            = google_container_cluster.lab.id
  location           = var.zone
  initial_node_count = 0

  # Scale from zero: the cluster autoscaler adds a node when a GPU pod is Pending and removes it
  # (after ~10 min idle) when nothing needs it.
  autoscaling {
    min_node_count = 0
    max_node_count = var.gpu_max_nodes
  }

  node_config {
    machine_type    = var.gpu_machine_type
    spot            = var.gpu_spot
    disk_size_gb    = var.gpu_disk_size_gb
    disk_type       = "pd-balanced"
    oauth_scopes    = local.oauth_scopes
    resource_labels = var.labels

    guest_accelerator {
      type  = var.gpu_type
      count = var.gpu_count

      # GKE installs the driver (no GPU Operator needed); DEFAULT or LATEST on COS.
      gpu_driver_installation_config {
        gpu_driver_version = var.gpu_driver_version
      }
    }

    taint {
      key    = local.gpu_taint.key
      value  = local.gpu_taint.value
      effect = local.gpu_taint.effect
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    gcfs_config {
      enabled = var.enable_image_streaming
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

resource "google_container_node_pool" "gpu_flex" {
  count              = var.enable_flex_start_pool ? 1 : 0
  name               = "l4-flex"
  cluster            = google_container_cluster.lab.id
  location           = var.zone
  initial_node_count = 0

  # Queued provisioning: nodes appear only for an accepted ProvisioningRequest (Kueue creates it),
  # all at once. Requirements (verify): total min 0, no reservation, auto-repair off.
  queued_provisioning {
    enabled = true
  }

  autoscaling {
    total_min_node_count = 0
    total_max_node_count = var.flex_max_nodes
    location_policy      = "ANY"
  }

  node_config {
    machine_type    = var.flex_machine_type
    flex_start      = true
    disk_size_gb    = var.gpu_disk_size_gb
    disk_type       = "pd-balanced"
    oauth_scopes    = local.oauth_scopes
    resource_labels = var.labels

    guest_accelerator {
      type  = var.gpu_type
      count = var.gpu_count

      gpu_driver_installation_config {
        gpu_driver_version = var.gpu_driver_version
      }
    }

    reservation_affinity {
      consume_reservation_type = "NO_RESERVATION"
    }

    taint {
      key    = local.gpu_taint.key
      value  = local.gpu_taint.value
      effect = local.gpu_taint.effect
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    gcfs_config {
      enabled = var.enable_image_streaming
    }
  }

  management {
    auto_repair  = false
    auto_upgrade = true
  }
}
