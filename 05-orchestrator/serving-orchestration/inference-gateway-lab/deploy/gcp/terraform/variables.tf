# ------------------------------------------------------------------------------------------
# Project and placement
# ------------------------------------------------------------------------------------------
variable "project_id" {
  description = "Project that hosts the cluster. GPUs need a paid billing account and L4 quota (GPUS_ALL_REGIONS and NVIDIA_L4_GPUS in the region)."
  type        = string
}

variable "region" {
  description = "Region of the VPC subnets and of the regional Application Load Balancer behind the Gateway."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Zone of the (zonal) cluster and its node pools. Pick one that offers NVIDIA L4 (g2 machines). # VERIFY: L4 zones in your region."
  type        = string
  default     = "us-central1-a"
}

variable "name" {
  description = "Prefix for the cluster, VPC and subnets."
  type        = string
  default     = "igw-lab"
}

variable "labels" {
  description = "Labels applied to the cluster, node pools and their VMs (cost attribution)."
  type        = map(string)
  default     = { app = "igw-lab", layer = "05-orchestrator" }
}

variable "enable_apis" {
  description = "Enable compute, container and monitoring APIs from Terraform (set false if your org manages APIs)."
  type        = bool
  default     = true
}

# ------------------------------------------------------------------------------------------
# Networking
# ------------------------------------------------------------------------------------------
variable "nodes_cidr" {
  description = "Primary range of the node subnet."
  type        = string
  default     = "10.10.0.0/20"
}

variable "pods_cidr" {
  description = "Secondary range for Pod IPs (VPC-native cluster)."
  type        = string
  default     = "10.20.0.0/16"
}

variable "services_cidr" {
  description = "Secondary range for Service IPs."
  type        = string
  default     = "10.30.0.0/20"
}

variable "proxy_only_cidr" {
  description = "Proxy-only subnet for the regional managed Envoy proxies of the Gateway (one per region and VPC)."
  type        = string
  default     = "10.129.0.0/23"
}

# ------------------------------------------------------------------------------------------
# Cluster
# ------------------------------------------------------------------------------------------
variable "release_channel" {
  description = "GKE release channel. The GKE-managed InferencePool v1 CRD needs GKE >= 1.34.0-gke.1626000. # VERIFY which channel has it in your region."
  type        = string
  default     = "RAPID"

  validation {
    condition     = contains(["RAPID", "REGULAR", "STABLE"], var.release_channel)
    error_message = "release_channel must be RAPID, REGULAR or STABLE."
  }
}

variable "monitoring_components" {
  description = "GKE components exporting metrics to Cloud Monitoring. Add \"DCGM\" for GPU metrics (Managed Prometheus must stay on)."
  type        = list(string)
  default     = ["SYSTEM_COMPONENTS"]
}

variable "system_machine_type" {
  description = "Machine type of the CPU pool that runs the EPP, the custom metrics adapter and system pods."
  type        = string
  default     = "e2-standard-4"
}

# ------------------------------------------------------------------------------------------
# GPU pool (model servers)
# ------------------------------------------------------------------------------------------
variable "gpu_machine_type" {
  description = "G2 machine with one NVIDIA L4 (24 GB). ~$0.70/hr on demand, 60-91% less as Spot (verify prices)."
  type        = string
  default     = "g2-standard-4"
}

variable "gpu_type" {
  description = "Accelerator type attached to each GPU node."
  type        = string
  default     = "nvidia-l4"
}

variable "gpus_per_node" {
  description = "GPUs per node (must match the machine type: g2-standard-4/8/12/16/32 have 1, g2-standard-24 has 2)."
  type        = number
  default     = 1
}

variable "gpu_spot" {
  description = "Use Spot VMs for the GPU pool (cheap, preemptible; fine for a lab)."
  type        = bool
  default     = true
}

variable "gpu_max_nodes" {
  description = "Upper bound of the GPU pool autoscaler (the pool starts at, and scales back to, 0 nodes)."
  type        = number
  default     = 2

  validation {
    condition     = var.gpu_max_nodes >= 1 && var.gpu_max_nodes <= 8
    error_message = "gpu_max_nodes must be between 1 and 8 for this lab."
  }
}
