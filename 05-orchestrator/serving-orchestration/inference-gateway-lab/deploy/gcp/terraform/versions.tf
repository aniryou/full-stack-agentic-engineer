# Provider and Terraform version pins for the inference-gateway lab (GKE Inference Gateway, T3).
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
