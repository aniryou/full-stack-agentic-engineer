locals {
  node_scopes = ["https://www.googleapis.com/auth/cloud-platform"] # access is governed by the SA's roles

  # Every GPU pool autoscales 0 -> max_nodes, so an idle cluster costs only the control plane and the
  # system node. GKE installs the driver and taints GPU nodes nvidia.com/gpu=present:NoSchedule (verify).
  # VERIFY: machine types and accelerators offered in var.zone; Spot availability for each GPU type.
  #
  #   pool       machine          GPUs/node  sharing                     used by (deploy/gke)
  #   l4         g2-standard-4    1 x L4     -                           01 smoke, 02 vectoradd
  #   l4x2       g2-standard-24   2 x L4     -                           03 nccl-tests (PCIe, no NVLink)
  #   l4-shared  g2-standard-4    1 x L4     time-sharing, N clients     04 time-sharing
  #   a100-mig   a2-highgpu-1g    1 x A100   MIG partitions (e.g. 7 x 1g.5gb)  05 mig
  gpu_pools = {
    l4 = {
      enabled              = true
      machine_type         = var.gpu_machine_type
      gpu_type             = var.gpu_type
      gpu_count            = 1
      spot                 = var.gpu_spot
      time_sharing_clients = null
      partition_size       = null
    }
    l4x2 = {
      enabled              = var.enable_multi_gpu_pool
      machine_type         = var.multi_gpu_machine_type
      gpu_type             = var.gpu_type
      gpu_count            = var.multi_gpu_count
      spot                 = var.gpu_spot
      time_sharing_clients = null
      partition_size       = null
    }
    l4-shared = {
      enabled              = var.enable_time_sharing_pool
      machine_type         = var.gpu_machine_type
      gpu_type             = var.gpu_type
      gpu_count            = 1
      spot                 = var.gpu_spot
      time_sharing_clients = var.time_sharing_clients
      partition_size       = null
    }
    a100-mig = {
      enabled              = var.enable_mig_pool
      machine_type         = var.mig_machine_type
      gpu_type             = var.mig_gpu_type
      gpu_count            = 1
      spot                 = var.mig_spot
      time_sharing_clients = null
      partition_size       = var.mig_partition_size
    }
  }
}

resource "google_container_node_pool" "system" {
  name       = "system"
  cluster    = google_container_cluster.lab.id
  location   = var.zone
  node_count = 1

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type    = var.system_machine_type
    image_type      = "COS_CONTAINERD"
    disk_size_gb    = 50
    service_account = google_service_account.nodes.email
    oauth_scopes    = local.node_scopes
    labels          = { "gpu-lab/pool" = "system" }
    resource_labels = var.labels

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}

resource "google_container_node_pool" "gpu" {
  for_each = { for name, pool in local.gpu_pools : name => pool if pool.enabled }

  name               = each.key
  cluster            = google_container_cluster.lab.id
  location           = var.zone
  initial_node_count = 0

  autoscaling {
    min_node_count = 0
    max_node_count = var.gpu_max_nodes
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type    = each.value.machine_type
    spot            = each.value.spot
    image_type      = "COS_CONTAINERD" # GKE-managed drivers and DCGM need Container-Optimized OS
    disk_size_gb    = 100              # CUDA devel images are several GB
    disk_type       = "pd-balanced"
    service_account = google_service_account.nodes.email
    oauth_scopes    = local.node_scopes
    labels          = { "gpu-lab/pool" = each.key }
    resource_labels = var.labels

    guest_accelerator {
      type               = each.value.gpu_type
      count              = each.value.gpu_count
      gpu_partition_size = each.value.partition_size

      gpu_driver_installation_config {
        gpu_driver_version = var.gpu_driver_version
      }

      dynamic "gpu_sharing_config" {
        for_each = each.value.time_sharing_clients == null ? [] : [each.value.time_sharing_clients]
        content {
          gpu_sharing_strategy       = "TIME_SHARING"
          max_shared_clients_per_gpu = gpu_sharing_config.value
        }
      }
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }

  depends_on = [google_container_node_pool.system]
}
