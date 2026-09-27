# Long-Running Agents on GCP — reference topology.
#
#   client/UI ──▶ Cloud Run: lra-api ──▶ Firestore (runs, effects)
#                     │  start/resume/cancel        ▲
#                     ▼                              │ checkpoints
#                Cloud Tasks queue ──OIDC──▶ Cloud Run: lra-worker ──▶ Vertex AI (Gemini)
#                     ▲                              │
#   Cloud Scheduler ──┘ (reap every 2 min)           ▼
#                                            Pub/Sub agent-events ──▶ push sub / BigQuery / DLQ
#   Cloud Workflows (declarative flows) ──OIDC──▶ lra-api
#
# Apply:  terraform init && terraform apply -var project_id=... -var region=asia-southeast1 -var image=...
# Not applied in this repo's CI; review resource names/quotas for your org.

terraform {
  required_version = ">= 1.6"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.30"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  services = [
    "run.googleapis.com",
    "cloudtasks.googleapis.com",
    "firestore.googleapis.com",
    "pubsub.googleapis.com",
    "cloudscheduler.googleapis.com",
    "workflows.googleapis.com",
    "aiplatform.googleapis.com",
    "cloudtrace.googleapis.com",
    "logging.googleapis.com",
    "artifactregistry.googleapis.com",
    "iamcredentials.googleapis.com",
  ]
}

resource "google_project_service" "apis" {
  for_each           = toset(local.services)
  service            = each.value
  disable_on_destroy = false
}

# ----------------------------------------------------------------- identities
resource "google_service_account" "api" {
  account_id   = "lra-api"
  display_name = "LRA control-plane API"
}

resource "google_service_account" "worker" {
  account_id   = "lra-worker"
  display_name = "LRA step worker"
}

resource "google_service_account" "tasks" {
  account_id   = "lra-tasks"
  display_name = "Identity Cloud Tasks/Scheduler use to call the worker"
}

resource "google_service_account" "workflow" {
  account_id   = "lra-workflow"
  display_name = "Cloud Workflows executions"
}

# Least privilege: api+worker read/write Firestore, publish Pub/Sub, call Vertex AI.
resource "google_project_iam_member" "runtime_roles" {
  for_each = {
    "api-datastore"    = { sa = google_service_account.api.email, role = "roles/datastore.user" }
    "api-pubsub"       = { sa = google_service_account.api.email, role = "roles/pubsub.publisher" }
    "api-tasks"        = { sa = google_service_account.api.email, role = "roles/cloudtasks.enqueuer" }
    "api-vertex"       = { sa = google_service_account.api.email, role = "roles/aiplatform.user" }
    "worker-datastore" = { sa = google_service_account.worker.email, role = "roles/datastore.user" }
    "worker-pubsub"    = { sa = google_service_account.worker.email, role = "roles/pubsub.publisher" }
    "worker-tasks"     = { sa = google_service_account.worker.email, role = "roles/cloudtasks.enqueuer" }
    "worker-vertex"    = { sa = google_service_account.worker.email, role = "roles/aiplatform.user" }
    "worker-trace"     = { sa = google_service_account.worker.email, role = "roles/cloudtrace.agent" }
    "api-trace"        = { sa = google_service_account.api.email, role = "roles/cloudtrace.agent" }
  }
  project = var.project_id
  role    = each.value.role
  member  = "serviceAccount:${each.value.sa}"
}

# Enqueuers must be allowed to mint OIDC tokens *as* the tasks SA.
resource "google_service_account_iam_member" "act_as_tasks_sa" {
  for_each           = toset([google_service_account.api.email, google_service_account.worker.email])
  service_account_id = google_service_account.tasks.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${each.value}"
}

# ----------------------------------------------------------------- firestore
resource "google_firestore_database" "runs" {
  name                        = var.firestore_database
  location_id                 = var.firestore_location
  type                        = "FIRESTORE_NATIVE"
  delete_protection_state     = "DELETE_PROTECTION_ENABLED"
  deletion_policy             = "ABANDON"
  point_in_time_recovery_enablement = "POINT_IN_TIME_RECOVERY_ENABLED"
  depends_on                  = [google_project_service.apis]
}

# The reaper queries by status; add a composite index if you also filter by lease expiry.
resource "google_firestore_index" "runs_status_updated" {
  database   = google_firestore_database.runs.name
  collection = "agent_runs"
  fields {
    field_path = "status"
    order      = "ASCENDING"
  }
  fields {
    field_path = "updated_at"
    order      = "ASCENDING"
  }
}

# ---------------------------------------------------------------- cloud tasks
resource "google_cloud_tasks_queue" "steps" {
  name     = "agent-steps"
  location = var.region

  rate_limits {
    max_dispatches_per_second = 50   # global step throughput cap
    max_concurrent_dispatches = 200  # in-flight steps across all workers
  }

  # Only `lease-held` (503) and infrastructure failures reach this policy;
  # the engine schedules its own step retries with explicit delays.
  retry_config {
    max_attempts       = 20
    min_backoff        = "2s"
    max_backoff        = "300s"
    max_doublings      = 6
    max_retry_duration = "0s" # unlimited: a parked run must eventually be re-driven
  }

  stackdriver_logging_config {
    sampling_ratio = 1.0
  }
  depends_on = [google_project_service.apis]
}

# -------------------------------------------------------------------- pub/sub
resource "google_pubsub_topic" "events" {
  name = "agent-events"
  message_retention_duration = "604800s" # 7 days: replayable for backfills
}

resource "google_pubsub_topic" "alerts" {
  name = "agent-alerts"
}

resource "google_pubsub_topic" "events_dlq" {
  name = "agent-events-dlq"
}

# Push subscription to the worker with a dead-letter policy (poison messages
# stop retrying after 5 attempts and land in the DLQ for inspection).
resource "google_pubsub_subscription" "events_push" {
  name  = "agent-events-worker"
  topic = google_pubsub_topic.events.name
  ack_deadline_seconds       = 30
  enable_message_ordering    = true
  message_retention_duration = "604800s"

  push_config {
    push_endpoint = "${google_cloud_run_v2_service.worker.uri}/pubsub/push"
    oidc_token {
      service_account_email = google_service_account.tasks.email
      audience              = google_cloud_run_v2_service.worker.uri
    }
  }

  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.events_dlq.id
    max_delivery_attempts = 5
  }

  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }
}

# Pub/Sub's service agent needs to publish to the DLQ and ack on the source.
data "google_project" "this" {}

resource "google_pubsub_topic_iam_member" "dlq_publisher" {
  topic  = google_pubsub_topic.events_dlq.name
  role   = "roles/pubsub.publisher"
  member = "serviceAccount:service-${data.google_project.this.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}

resource "google_pubsub_subscription_iam_member" "dlq_subscriber" {
  subscription = google_pubsub_subscription.events_push.name
  role         = "roles/pubsub.subscriber"
  member       = "serviceAccount:service-${data.google_project.this.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}

# Optional analytics sink: agent-events -> BigQuery (cost per run, step latency).
resource "google_bigquery_dataset" "agent" {
  count      = var.enable_bigquery_sink ? 1 : 0
  dataset_id = "agent_events"
  location   = var.bq_location
}

resource "google_pubsub_subscription" "events_bq" {
  count = var.enable_bigquery_sink ? 1 : 0
  name  = "agent-events-bq"
  topic = google_pubsub_topic.events.name
  bigquery_config {
    table          = "${var.project_id}.${google_bigquery_dataset.agent[0].dataset_id}.events"
    write_metadata = true
  }
}

# ------------------------------------------------------------------ cloud run
resource "google_cloud_run_v2_service" "worker" {
  name     = "lra-worker"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_INTERNAL_ONLY" # Cloud Tasks / Scheduler / Pub/Sub are internal callers

  template {
    service_account = google_service_account.worker.email
    timeout         = "1800s" # == Cloud Tasks dispatch deadline ceiling
    max_instance_request_concurrency = 8
    scaling {
      min_instance_count = 0   # sleep is free
      max_instance_count = 50
    }
    containers {
      image = var.image
      env {
        name  = "LRA_SERVICE"
        value = "worker"
      }
      dynamic "env" {
        for_each = local.common_env
        content {
          name  = env.key
          value = env.value
        }
      }
      resources {
        limits = { cpu = "1", memory = "1Gi" }
        cpu_idle = true
      }
    }
  }
  depends_on = [google_project_service.apis]
}

resource "google_cloud_run_v2_service" "api" {
  name     = "lra-api"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER" # front with LB + IAP or API Gateway

  template {
    service_account = google_service_account.api.email
    timeout         = "60s"
    scaling {
      min_instance_count = 0
      max_instance_count = 10
    }
    containers {
      image = var.image
      env {
        name  = "LRA_SERVICE"
        value = "api"
      }
      dynamic "env" {
        for_each = local.common_env
        content {
          name  = env.key
          value = env.value
        }
      }
    }
  }
  depends_on = [google_project_service.apis]
}

locals {
  common_env = {
    LRA_BACKEND          = "gcp"
    GOOGLE_CLOUD_PROJECT = var.project_id
    LRA_LOCATION         = var.region
    LRA_TASKS_QUEUE      = google_cloud_tasks_queue.steps.name
    LRA_TASKS_SA         = google_service_account.tasks.email
    LRA_WORKER_URL       = "https://lra-worker-${data.google_project.this.number}.${var.region}.run.app" # deterministic URL; avoids a cycle with the worker resource
    LRA_FIRESTORE_DB     = google_firestore_database.runs.name
    LRA_GEMINI_MODEL     = var.gemini_model
    LRA_GEMINI_LOCATION  = var.gemini_location
    LRA_LEASE_TTL_S      = "120"
  }
}

# Only the tasks SA (used by Cloud Tasks, Scheduler and the Pub/Sub push) may invoke the worker.
resource "google_cloud_run_v2_service_iam_member" "worker_invoker" {
  name     = google_cloud_run_v2_service.worker.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.tasks.email}"
}

resource "google_cloud_run_v2_service_iam_member" "api_invoker_workflow" {
  name     = google_cloud_run_v2_service.api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.workflow.email}"
}

# -------------------------------------------------------------- reaper cron
resource "google_cloud_scheduler_job" "reaper" {
  name             = "lra-reaper"
  schedule         = "*/2 * * * *"
  time_zone        = "Etc/UTC"
  attempt_deadline = "120s"
  region           = var.region

  http_target {
    http_method = "POST"
    uri         = "${google_cloud_run_v2_service.worker.uri}/internal/reap"
    headers     = { "Content-Type" = "application/json", "X-CloudTasks-QueueName" = "scheduler" }
    oidc_token {
      service_account_email = google_service_account.tasks.email
      audience              = google_cloud_run_v2_service.worker.uri
    }
  }
}

# ---------------------------------------------------------------- workflows
resource "google_workflows_workflow" "research_approval" {
  name            = "research-approval"
  region          = var.region
  service_account = google_service_account.workflow.id
  source_contents = file("${path.module}/../../workflows/research_approval.yaml")
  user_env_vars = {
    LRA_API_URL = google_cloud_run_v2_service.api.uri
  }
  depends_on = [google_project_service.apis]
}
