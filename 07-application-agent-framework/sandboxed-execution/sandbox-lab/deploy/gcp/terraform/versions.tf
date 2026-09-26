# Provider and Terraform pins for the sandbox-lab GKE cluster (validated offline with google provider 8.4.0).

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

  # Every labelable resource gets these, so the billing export can attribute the lab's spend.
  default_labels = var.labels
}
