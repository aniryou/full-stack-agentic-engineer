# Provider and Terraform pins for the k8s-gpu-lab GKE cluster (validated with google provider 8.4.0).

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

  # Every labelable resource gets these: `gcloud billing` / the billing export can then attribute
  # the lab's spend (GPU VMs included) to one label.
  default_labels = var.labels
}
