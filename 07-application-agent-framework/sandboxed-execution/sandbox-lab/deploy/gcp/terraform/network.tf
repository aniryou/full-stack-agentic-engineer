# A dedicated VPC. Nodes are private (no external IPs) and, by default, there is no Cloud NAT: nothing in
# the cluster has a route to the internet. Private Google Access lets nodes pull images from Artifact
# Registry and write logs and metrics (verify that it covers *.pkg.dev for your setup).

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

# Optional: a route to the internet for the egress proxy's allowlisted hosts. It applies to the whole
# subnet, so from here on the sandbox pods are kept off the internet by NetworkPolicy (Dataplane V2) and
# gVisor, not by the absence of a route.
resource "google_compute_router" "nat" {
  count   = var.enable_nat ? 1 : 0
  name    = "${var.cluster_name}-router"
  region  = var.region
  network = google_compute_network.lab.id
}

resource "google_compute_router_nat" "nat" {
  count                              = var.enable_nat ? 1 : 0
  name                               = "${var.cluster_name}-nat"
  router                             = google_compute_router.nat[0].name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}
