# A private network for one VM: SSH only through Identity-Aware Proxy, egress through an
# ephemeral external IP (default) or Cloud NAT.

resource "google_compute_network" "bench" {
  name                    = "${local.name}-net"
  auto_create_subnetworks = false
  depends_on              = [google_project_service.apis]
}

resource "google_compute_subnetwork" "bench" {
  name                     = "${local.name}-subnet"
  ip_cidr_range            = var.subnet_cidr
  region                   = var.region
  network                  = google_compute_network.bench.id
  private_ip_google_access = true
}

resource "google_compute_firewall" "iap_ssh" {
  name          = "${local.name}-allow-iap-ssh"
  network       = google_compute_network.bench.name
  direction     = "INGRESS"
  source_ranges = ["35.235.240.0/20"] # Google's IAP TCP-forwarding range
  target_tags   = [var.name]

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

resource "google_compute_router" "nat" {
  count   = var.assign_public_ip ? 0 : 1
  name    = "${local.name}-router"
  region  = var.region
  network = google_compute_network.bench.id
}

resource "google_compute_router_nat" "nat" {
  count                              = var.assign_public_ip ? 0 : 1
  name                               = "${local.name}-nat"
  router                             = google_compute_router.nat[0].name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}
