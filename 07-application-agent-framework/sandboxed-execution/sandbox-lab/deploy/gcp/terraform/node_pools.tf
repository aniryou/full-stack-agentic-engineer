# Two pools. The system pool runs GKE's components, the egress proxy and the stand-in upstream. The
# sandbox pool runs gVisor: GKE Sandbox needs a second pool (the first cannot be sandboxed), COS with
# containerd, and adds the label and taint sandbox.gke.io/runtime=gvisor itself (verify), which is why
# no taint is declared below - the RuntimeClass "gvisor" that GKE creates carries the matching
# nodeSelector and toleration, so a pod needs only `runtimeClassName: gvisor`.

locals {
  gke_sandbox_taint = "sandbox.gke.io/runtime=gvisor:NoSchedule" # added by GKE (VERIFY: kubectl describe node)
}

resource "google_container_node_pool" "system" {
  name       = "system"
  cluster    = google_container_cluster.lab.id
  location   = var.zone
  node_count = var.system_node_count

  node_config {
    machine_type    = var.system_machine_type
    image_type      = "COS_CONTAINERD"
    service_account = google_service_account.nodes.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

resource "google_container_node_pool" "sandbox" {
  name     = "gvisor"
  cluster  = google_container_cluster.lab.id
  location = var.zone

  autoscaling {
    min_node_count = 0
    max_node_count = var.sandbox_max_nodes
  }

  node_config {
    machine_type    = var.sandbox_machine_type
    image_type      = "COS_CONTAINERD"
    spot            = var.sandbox_spot
    service_account = google_service_account.nodes.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    # GKE Sandbox. The provider accepts only the upper-case value (gcloud's flag is `--sandbox type=gvisor`).
    sandbox_config {
      type = "GVISOR"
    }

    # Pods get their KSA's Workload Identity from the GKE metadata server, never the node's account.
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    kubelet_config {
      pod_pids_limit = var.pod_pids_limit
    }

    labels = {
      "sandboxlab-pool" = "gvisor"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}
