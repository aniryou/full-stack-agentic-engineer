# Minimal infra for reference architecture A (durable loop on Cloud Run) +
# the Cloud Workflows variants. Apply with:
#   terraform init && terraform apply -var project_id=... -var region=asia-southeast1
# Cloud Run is deployed by infra/deploy.sh (source deploy) so this file only
# creates the queue/topics/schedule/IAM around it.

terraform {
  required_version = ">= 1.5"
  required_providers {
    google = { source = "hashicorp/google", version = "~> 6.0" }
  }
}

variable "project_id" {
  type = string
}
variable "region" {
  type    = string
  default = "asia-southeast1"
}
variable "service_name" {
  type    = string
  default = "lragents"
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  apis = ["run.googleapis.com", "cloudtasks.googleapis.com", "pubsub.googleapis.com", "cloudscheduler.googleapis.com",
          "workflows.googleapis.com", "firestore.googleapis.com", "aiplatform.googleapis.com", "eventarc.googleapis.com",
          "cloudtrace.googleapis.com", "iap.googleapis.com"]
}

resource "google_project_service" "apis" {
  for_each           = toset(local.apis)
  service            = each.key
  disable_on_destroy = false
}

# ---------------------------------------------------------------- identities
resource "google_service_account" "run" {
  account_id   = "sa-agent-run"
  display_name = "Cloud Run runtime for the agent service"
}
resource "google_service_account" "invoker" {
  account_id   = "sa-tasks-invoker"
  display_name = "Identity Cloud Tasks / Pub/Sub push use to call the service"
}
resource "google_service_account" "workflows" {
  account_id   = "sa-workflows"
  display_name = "Cloud Workflows executions"
}

resource "google_project_iam_member" "run_roles" {
  for_each = toset(["roles/datastore.user", "roles/cloudtasks.enqueuer", "roles/pubsub.publisher",
                    "roles/aiplatform.user", "roles/secretmanager.secretAccessor", "roles/cloudtrace.agent", "roles/logging.logWriter"])
  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.run.email}"
}
# The runtime SA must be allowed to mint OIDC tokens as the invoker SA when creating tasks.
resource "google_service_account_iam_member" "run_acts_as_invoker" {
  service_account_id = google_service_account.invoker.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.run.email}"
}

# ---------------------------------------------------------------- Firestore
resource "google_firestore_database" "db" {
  name        = "(default)"
  location_id = var.region
  type        = "FIRESTORE_NATIVE"
  depends_on  = [google_project_service.apis]
}
resource "google_firestore_field" "idem_ttl" {
  database   = google_firestore_database.db.name
  collection = "idempotency_keys"
  field      = "expires_at"
  ttl_config {}
}

# ---------------------------------------------------------------- Cloud Tasks
resource "google_cloud_tasks_queue" "steps" {
  name     = "agent-steps"
  location = var.region
  rate_limits {
    max_dispatches_per_second = 50
    max_concurrent_dispatches = 20      # protects Cloud Run; raise with capacity
  }
  retry_config {
    max_attempts  = 20
    min_backoff   = "2s"
    max_backoff   = "300s"
    max_doublings = 6
  }
  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------- Pub/Sub
resource "google_pubsub_topic" "events" {
  name = "agent-events"
}
resource "google_pubsub_topic" "ticks" {
  name = "agent-ticks"
}
resource "google_pubsub_topic" "dlq" {
  name = "agent-dlq"
}

# Push to the service with an OIDC token; the service verifies audience + caller email.
resource "google_pubsub_subscription" "events_push" {
  name  = "agent-events-push"
  topic = google_pubsub_topic.events.name
  ack_deadline_seconds = 60
  push_config {
    push_endpoint = "${var.service_url}/internal/pubsub/push"
    oidc_token { service_account_email = google_service_account.invoker.email }
  }
  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.dlq.id
    max_delivery_attempts = 10
  }
  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }
}
resource "google_pubsub_subscription" "ticks_push" {
  name  = "agent-ticks-push"
  topic = google_pubsub_topic.ticks.name
  push_config {
    push_endpoint = "${var.service_url}/internal/scheduler/tick"
    oidc_token { service_account_email = google_service_account.invoker.email }
  }
}
variable "service_url" {
  type        = string
  description = "Cloud Run service URL (known after the first deploy; re-apply to wire push subscriptions)"
  default     = "https://example.invalid"
}

# ---------------------------------------------------------------- Scheduler
resource "google_cloud_scheduler_job" "tick" {
  name      = "agent-tick"
  schedule  = "*/5 * * * *"
  time_zone = "Asia/Singapore"
  pubsub_target {
    topic_name = google_pubsub_topic.ticks.id
    data       = base64encode("{\"kind\":\"tick\"}")
  }
}

# ---------------------------------------------------------------- Workflows
resource "google_workflows_workflow" "hitl" {
  name            = "hitl-approval"
  region          = var.region
  service_account = google_service_account.workflows.id
  source_contents = file("${path.module}/../workflows/hitl_approval.yaml")
}
resource "google_workflows_workflow" "fan" {
  name            = "fan-out-fan-in"
  region          = var.region
  service_account = google_service_account.workflows.id
  source_contents = file("${path.module}/../workflows/fan_out_fan_in.yaml")
}

# ---------------------------------------------------------------- outputs
output "run_service_account" {
  value = google_service_account.run.email
}
output "invoker_service_account" {
  value = google_service_account.invoker.email
}
output "workflows_service_account" {
  value = google_service_account.workflows.email
}
output "tasks_queue" {
  value = google_cloud_tasks_queue.steps.name
}
