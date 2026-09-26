# ------------------------------------------------------------------------------------------
# Project and placement
# ------------------------------------------------------------------------------------------

variable "project_id" {
  description = "Project that hosts the cluster. It needs billing (GPUs are not available on a Free Trial account) and GPU quota for the chosen type."
  type        = string
}

variable "region" {
  description = "Region for the network and the weights bucket."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Zone of the (zonal) cluster and its node pools. Pick one that offers var.gpu_type (L4: several us-central1 zones - verify)."
  type        = string
  default     = "us-central1-a"
}

variable "cluster_name" {
  description = "GKE cluster name; also prefixes the network, subnet and bucket names."
  type        = string
  default     = "gpu-lab"
}

variable "release_channel" {
  description = "GKE release channel. Flex-start needs GKE >= 1.32.2-gke.1652000 and default driver auto-install >= 1.32.2-gke.1297000 (verify); REGULAR is past both."
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
    app     = "k8s-gpu-lab"
    purpose = "learning"
  }
}

variable "deletion_protection" {
  description = "Protect the cluster from `terraform destroy`. Off for a lab you tear down after each session."
  type        = bool
  default     = false
}

# ------------------------------------------------------------------------------------------
# Networking (a dedicated VPC so the lab never depends on, or disturbs, the default network)
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
  description = "Secondary range for Service ClusterIPs."
  type        = string
  default     = "10.30.0.0/20"
}

# ------------------------------------------------------------------------------------------
# System pool (CPU): kube-system, Kueue, JobSet and other controllers live here
# ------------------------------------------------------------------------------------------

variable "system_machine_type" {
  description = "Machine type of the always-on system pool."
  type        = string
  default     = "e2-standard-4"
}

variable "system_node_count" {
  description = "Nodes in the system pool."
  type        = number
  default     = 1
}

# ------------------------------------------------------------------------------------------
# GPU pool: L4 on Spot, autoscaling from zero, driver installed by GKE
# ------------------------------------------------------------------------------------------

variable "gpu_machine_type" {
  description = "Machine type of the GPU pool (g2-standard-4 = 1 x L4, 4 vCPU, 16 GB)."
  type        = string
  default     = "g2-standard-4"
}

variable "gpu_type" {
  description = "GKE accelerator type; becomes the cloud.google.com/gke-accelerator node label."
  type        = string
  default     = "nvidia-l4"
}

variable "gpu_count" {
  description = "GPUs per node (must match what the machine type carries)."
  type        = number
  default     = 1
}

variable "gpu_driver_version" {
  description = "GKE-managed NVIDIA driver: DEFAULT, LATEST (COS only) or INSTALLATION_DISABLED (bring your own, e.g. the GPU Operator). LATEST because the lab's vLLM v0.30.0 image is a CUDA 13.0 build that needs driver >= 580; DEFAULT may be an older branch (verify)."
  type        = string
  default     = "LATEST"

  validation {
    condition     = contains(["DEFAULT", "LATEST", "INSTALLATION_DISABLED"], var.gpu_driver_version)
    error_message = "gpu_driver_version must be DEFAULT, LATEST or INSTALLATION_DISABLED."
  }
}

variable "gpu_spot" {
  description = "Run the GPU pool on Spot VMs (60-91% cheaper; can be reclaimed with ~30 s notice)."
  type        = bool
  default     = true
}

variable "gpu_max_nodes" {
  description = "Upper bound for the GPU pool's autoscaler (the pool idles at zero nodes)."
  type        = number
  default     = 2
}

variable "gpu_disk_size_gb" {
  description = "Boot disk per GPU node; serving images (vLLM) are large."
  type        = number
  default     = 100
}

# ------------------------------------------------------------------------------------------
# Optional DWS flex-start pool with queued provisioning (used through Kueue's ProvisioningRequest check)
# ------------------------------------------------------------------------------------------

variable "enable_flex_start_pool" {
  description = "Create the l4-flex pool: flex-start VMs, queued provisioning (all-or-nothing), max run 7 days. Off by default."
  type        = bool
  default     = false
}

variable "flex_machine_type" {
  description = "Machine type of the flex-start pool."
  type        = string
  default     = "g2-standard-4"
}

variable "flex_max_nodes" {
  description = "Upper bound on nodes DWS may provision at once in the flex-start pool."
  type        = number
  default     = 2
}

# ------------------------------------------------------------------------------------------
# Optional time-sharing pool: one physical L4 advertised as several nvidia.com/gpu (primer section 9)
# ------------------------------------------------------------------------------------------

variable "enable_time_sharing_pool" {
  description = "Create the l4-shared pool: Spot L4 nodes with GPU time-sharing (deploy/gke/50-time-sharing-l4.yaml). Off by default."
  type        = bool
  default     = false
}

variable "max_shared_clients_per_gpu" {
  description = "Pods that may share one physical GPU on the l4-shared pool; the node advertises gpu_count x this many nvidia.com/gpu."
  type        = number
  default     = 4

  validation {
    condition     = var.max_shared_clients_per_gpu >= 2 && var.max_shared_clients_per_gpu <= 48
    error_message = "max_shared_clients_per_gpu must be between 2 and 48 (GKE's limit - verify)."
  }
}

# ------------------------------------------------------------------------------------------
# Startup latency and weights
# ------------------------------------------------------------------------------------------

variable "enable_image_streaming" {
  description = "Image streaming (GCFS): containers start before the whole image is pulled (Artifact Registry images)."
  type        = bool
  default     = true
}

variable "enable_gcs_fuse" {
  description = "Cloud Storage FUSE CSI driver add-on: mount a bucket of model weights into pods."
  type        = bool
  default     = true
}

variable "create_weights_bucket" {
  description = "Create a bucket for model weights and let the serving KSA read it via Workload Identity."
  type        = bool
  default     = true
}

variable "weights_bucket_location" {
  description = "Bucket location; keep it in the cluster's region for bandwidth and egress cost."
  type        = string
  default     = "US-CENTRAL1"
}

variable "weights_namespace" {
  description = "Namespace of the serving pods that read the weights (deploy/gke/40-serving-vllm-gcsfuse.yaml)."
  type        = string
  default     = "serving"
}

variable "weights_ksa" {
  description = "Kubernetes ServiceAccount of those pods; it gets roles/storage.objectViewer on the bucket."
  type        = string
  default     = "model-reader"
}
