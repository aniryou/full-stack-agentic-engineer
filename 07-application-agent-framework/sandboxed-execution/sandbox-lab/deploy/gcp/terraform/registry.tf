# Artifact Registry for the sandbox image (a copy of python:3.12-slim: deploy/gcp/mirror-image.sh).
# Private nodes without NAT cannot pull from Docker Hub.

resource "google_artifact_registry_repository" "sandbox" {
  repository_id = "sandbox"
  location      = var.region
  format        = "DOCKER"
  description   = "Images for sandbox-lab: the sandbox runtime, the egress proxy and the stand-in upstream."

  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"
    most_recent_versions {
      keep_count = 5
    }
  }

  depends_on = [google_project_service.apis]
}
