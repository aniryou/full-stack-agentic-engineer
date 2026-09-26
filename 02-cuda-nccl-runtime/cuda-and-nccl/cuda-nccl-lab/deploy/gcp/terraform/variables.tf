# ------------------------------------------------------------------------------------------------
# Project and location
# ------------------------------------------------------------------------------------------------
variable "project_id" {
  description = "Project to create the cluster in (billing enabled; GPUs are not available on a Free Trial billing account)."
  type        = string
}

variable "region" {
  description = "Region of the zone below; used for the subnetwork."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Zone for the zonal cluster and all node pools. Must offer L4 (and A100 if enable_mig_pool)."
  type        = string
  default     = "us-central1-a" # VERIFY: gcloud compute accelerator-types list --filter="zone:us-central1-a"
}

variable "cluster_name" {
  description = "GKE cluster name."
  type        = string
  default     = "gpu-lab"
}

variable "release_channel" {
  description = "GKE release channel. GPU driver auto-install and the DCGM package need a recent GKE version."
  type        = string
  default     = "REGULAR"
}

variable "labels" {
  description = "Resource labels on the cluster and nodes (cost attribution)."
  type        = map(string)
  default     = { app = "cuda-nccl-lab", managed-by = "terraform" }
}

# ------------------------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------------------------
variable "subnet_cidr" {
  description = "Primary range for nodes."
  type        = string
  default     = "10.10.0.0/20"
}

variable "pods_cidr" {
  description = "Secondary range for Pods (VPC-native)."
  type        = string
  default     = "10.20.0.0/16"
}

variable "services_cidr" {
  description = "Secondary range for Services."
  type        = string
  default     = "10.30.0.0/20"
}

# ------------------------------------------------------------------------------------------------
# Node pools
# ------------------------------------------------------------------------------------------------
variable "system_machine_type" {
  description = "CPU pool for kube-system, the DCGM exporter's control side and Managed Prometheus collectors."
  type        = string
  default     = "e2-standard-4"
}

variable "gpu_driver_version" {
  description = "GKE-managed NVIDIA driver: DEFAULT, LATEST (COS only) or INSTALLATION_DISABLED (bring your own, e.g. the GPU Operator)."
  type        = string
  default     = "DEFAULT"

  validation {
    condition     = contains(["DEFAULT", "LATEST", "INSTALLATION_DISABLED"], var.gpu_driver_version)
    error_message = "gpu_driver_version must be DEFAULT, LATEST or INSTALLATION_DISABLED."
  }
}

variable "gpu_machine_type" {
  description = "Single-GPU pool (smoke test, vectoradd, kernels): g2-standard-4 = 1 x L4, 4 vCPU, 16 GB."
  type        = string
  default     = "g2-standard-4"
}

variable "gpu_type" {
  description = "Accelerator type for the L4 pools."
  type        = string
  default     = "nvidia-l4"
}

variable "gpu_max_nodes" {
  description = "Autoscaling ceiling of each GPU pool (the floor is always 0: no GPU pod, no GPU bill)."
  type        = number
  default     = 2
}

variable "gpu_spot" {
  description = "Spot VMs for the L4 pools: much cheaper, can be preempted — fine for a lab."
  type        = bool
  default     = true
}

variable "enable_multi_gpu_pool" {
  description = "A 2 x L4 pool (g2-standard-24) for nccl-tests (deploy/gke/03-nccl-tests-2gpu.yaml). Costs nothing until a pod needs it."
  type        = bool
  default     = true
}

variable "multi_gpu_machine_type" {
  description = "Machine type of the multi-GPU pool."
  type        = string
  default     = "g2-standard-24"
}

variable "multi_gpu_count" {
  description = "GPUs per node in the multi-GPU pool (must match the machine type)."
  type        = number
  default     = 2
}

variable "enable_time_sharing_pool" {
  description = "An L4 pool with GPU time-sharing (deploy/gke/04-time-sharing.yaml)."
  type        = bool
  default     = false
}

variable "time_sharing_clients" {
  description = "max_shared_clients_per_gpu: each physical GPU is advertised as this many nvidia.com/gpu."
  type        = number
  default     = 2
}

variable "enable_mig_pool" {
  description = "An A100 pool partitioned with MIG (deploy/gke/05-mig.yaml). Off by default: A100 quota is hard to get."
  type        = bool
  default     = false
}

variable "mig_machine_type" {
  description = "Machine type for the MIG pool."
  type        = string
  default     = "a2-highgpu-1g"
}

variable "mig_gpu_type" {
  description = "Accelerator type for the MIG pool."
  type        = string
  default     = "nvidia-tesla-a100"
}

variable "mig_partition_size" {
  description = "MIG profile for every GPU in the pool, e.g. 1g.5gb (7 per A100 40GB)."
  type        = string
  default     = "1g.5gb" # VERIFY: partition sizes GKE supports for your GPU
}

variable "mig_spot" {
  description = "Spot VMs for the MIG pool."
  type        = bool
  default     = true
}

variable "enable_dcgm" {
  description = "GKE-managed DCGM exporter; metrics land in Cloud Monitoring through Managed Prometheus."
  type        = bool
  default     = true
}
