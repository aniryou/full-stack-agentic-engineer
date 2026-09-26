# A dedicated VPC: a node subnet with secondary ranges (VPC-native cluster), plus the proxy-only
# subnet that regional Application Load Balancers — the data plane of the GKE Gateway class
# gke-l7-regional-external-managed — need for their managed Envoy proxies.
resource "google_compute_network" "vpc" {
  name                    = "${var.name}-vpc"
  auto_create_subnetworks = false

  depends_on = [google_project_service.apis]
}

resource "google_compute_subnetwork" "nodes" {
  name                     = "${var.name}-nodes"
  region                   = var.region
  network                  = google_compute_network.vpc.id
  ip_cidr_range            = var.nodes_cidr
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

resource "google_compute_subnetwork" "proxy_only" {
  name          = "${var.name}-proxy-only"
  region        = var.region
  network       = google_compute_network.vpc.id
  ip_cidr_range = var.proxy_only_cidr
  purpose       = "REGIONAL_MANAGED_PROXY"
  role          = "ACTIVE"
}
