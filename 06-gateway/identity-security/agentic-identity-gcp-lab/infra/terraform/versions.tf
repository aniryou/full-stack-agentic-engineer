
# Provider and Terraform version pins for the agentsec lab.
#
# NOTE on the Terraform floor: the design brief asked for >= 1.9, but this configuration uses
# *write-only* arguments (`client_secret_wo` on the Auth Manager provider and `secret_data_wo`
# on the Secret Manager version) so that OAuth client secrets and demo secrets never land in
# Terraform state. Write-only arguments require Terraform >= 1.11 — see primer §5 ("Secrets").

terraform {
  required_version = ">= 1.11"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 8.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = ">= 8.0"
    }
  }
}

# `user_project_override` + `billing_project` make quota/billing attribution explicit for the
# APIs that require a quota project when called with user credentials (Access Context Manager,
# IAM v3 policy bindings, Model Armor).
provider "google" {
  project               = var.project_id
  region                = var.region
  user_project_override = true
  billing_project       = var.project_id
}

provider "google-beta" {
  project               = var.project_id
  region                = var.region
  user_project_override = true
  billing_project       = var.project_id
}
