# ------------------------------------------------------------------------------------------
# Project and placement
# ------------------------------------------------------------------------------------------

variable "project_id" {
  description = "Project that hosts the cluster (needs billing; no GPUs are used)."
  type        = string
}

variable "region" {
  description = "Region of the network, the Artifact Registry repository and the optional Cloud NAT."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Zone of the zonal cluster and its node pools (one zonal cluster's management fee is covered by the GKE free tier - verify)."
  type        = string
  default     = "us-central1-a"
}

variable "cluster_name" {
  description = "GKE cluster name; also prefixes the network, subnet and repository names."
  type        = string
  default     = "sandbox-lab"
}

variable "release_channel" {
  description = "GKE release channel."
  type        = string
  default     = "REGULAR"

  validation {
    condition     = contains(["RAPID", "REGULAR", "STABLE"], var.release_channel)
    error_message = "release_channel must be RAPID, REGULAR or STABLE."
  }
}

variable "labels" {
  description = "Labels applied to every labelable resource (cost attribution)."
  type        = map(string)
  default = {
    app     = "sandbox-lab"
    purpose = "learning"
  }
}

variable "deletion_protection" {
  description = "Protect the cluster from `terraform destroy`. Off for a lab you tear down after each session."
  type        = bool
  default     = false
}

# ------------------------------------------------------------------------------------------
# Networking: private nodes, no route to the internet unless enable_nat
# ------------------------------------------------------------------------------------------

variable "subnet_cidr" {
  description = "Primary range of the node subnet."
  type        = string
  default     = "10.10.0.0/20"
}

variable "pods_cidr" {
  description = "Secondary range for pod IPs (VPC-native cluster)."
  type        = string
  default     = "10.20.0.0/16"
}

variable "services_cidr" {
  description = "Secondary range for Service ClusterIPs. The egress proxy's fixed ClusterIP (10.30.0.200 in deploy/gke) must be inside it."
  type        = string
  default     = "10.30.0.0/20"
}

variable "master_cidr" {
  description = "The /28 the private control plane peers into (must not overlap the ranges above)."
  type        = string
  default     = "172.16.0.32/28"
}

variable "authorized_networks" {
  description = "CIDRs allowed to reach the control plane's public endpoint (your IP /32). Empty = no restriction beyond Google auth."
  type        = list(string)
  default     = []
}

variable "enable_nat" {
  description = "Add Cloud Router + Cloud NAT so pods can reach the internet. Off: the only way out is Google APIs over Private Google Access; the egress proxy then reaches only in-cluster upstreams."
  type        = bool
  default     = false
}

# ------------------------------------------------------------------------------------------
# System pool: the proxy, the stand-in upstream and GKE's own components
# ------------------------------------------------------------------------------------------

variable "system_machine_type" {
  description = "Machine type of the always-on system pool."
  type        = string
  default     = "e2-standard-2"
}

variable "system_node_count" {
  description = "Nodes in the system pool."
  type        = number
  default     = 1
}

# ------------------------------------------------------------------------------------------
# Sandbox pool: GKE Sandbox (gVisor), Spot, autoscaling from zero
# ------------------------------------------------------------------------------------------

variable "sandbox_machine_type" {
  description = "Machine type of the gVisor pool (not an E2 shared-core type: GKE Sandbox does not support e2-micro/small/medium - verify)."
  type        = string
  default     = "e2-standard-2"
}

variable "sandbox_spot" {
  description = "Run the sandbox pool on Spot VMs (60-91% cheaper; reclaimable). Executions are short and retried by the caller, not by Kubernetes."
  type        = bool
  default     = true
}

variable "sandbox_max_nodes" {
  description = "Upper bound of the sandbox pool's autoscaler (the pool starts at zero)."
  type        = number
  default     = 3
}

variable "pod_pids_limit" {
  description = "kubelet podPidsLimit on the sandbox nodes: the only per-pod process limit Kubernetes has."
  type        = number
  default     = 128
}

variable "enable_agent_sandbox_addon" {
  description = "Enable GKE's Agent Sandbox add-on (kubernetes-sigs/agent-sandbox managed by GKE - verify availability and version in your release channel). deploy/gke/apply.sh can instead install the upstream controller."
  type        = bool
  default     = false
}
