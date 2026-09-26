# A dedicated VPC and subnet with secondary ranges for a VPC-native cluster. Nodes keep public IPs
# (no Cloud NAT to pay for); lock this down for anything beyond a lab.

resource "google_compute_network" "lab" {
  name                    = "${var.cluster_name}-net"
  auto_create_subnetworks = false
  depends_on              = [google_project_service.apis]
}

resource "google_compute_subnetwork" "lab" {
  name                     = "${var.cluster_name}-subnet"
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
