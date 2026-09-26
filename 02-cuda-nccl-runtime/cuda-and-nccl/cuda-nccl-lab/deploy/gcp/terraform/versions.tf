# A small, cheap GKE Standard cluster for the cuda-nccl lab: one zonal control plane, a CPU system
# pool, and GPU node pools that autoscale from zero (Spot by default), with GKE-managed NVIDIA
# drivers, DCGM metrics and Google Managed Prometheus. See ../README.md for cost and cleanup.

terraform {
  required_version = ">= 1.9"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 8.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
