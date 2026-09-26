# A dedicated VPC so the lab is self-contained and `terraform destroy` removes everything.
# Nodes keep public IPs (no Cloud NAT to pay for); the nccl-tests Job pulls from GitHub and apt.

resource "google_compute_network" "lab" {
  name                    = "${var.cluster_name}-vpc"
  auto_create_subnetworks = false
  depends_on              = [google_project_service.apis]
}

resource "google_compute_subnetwork" "lab" {
  name                     = "${var.cluster_name}-nodes"
  region                   = var.region
  network                  = google_compute_network.lab.id
  ip_cidr_range            = var.subnet_cidr
  private_ip_google_access = true

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = var.pods_cidr
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = var.services_cidr
  }
}
