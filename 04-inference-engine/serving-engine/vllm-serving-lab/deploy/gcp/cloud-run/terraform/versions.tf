# vLLM on Cloud Run with one NVIDIA L4 — provider and Terraform pins.
# Validated offline against hashicorp/google 8.4.0 (Sep 2026).

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
