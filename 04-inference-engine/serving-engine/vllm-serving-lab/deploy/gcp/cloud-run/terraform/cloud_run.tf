# The engine: `vllm serve` in one Cloud Run instance with one L4, scaling 0..max_instances.
#
# What each block maps to in the lab:
#   node_selector + limits["nvidia.com/gpu"]  one L4 (24 GB) per instance: sizing.size(..., "L4")
#   args                                      the same flags you tuned with tune.sweep at T0/T1
#   max_instance_request_concurrency          requests per instance: pick it from the batch that
#                                             meets your SLO (notebook 06), not from a default
#   scaling.min_instance_count = 0            scale to zero: idle costs nothing, a cold start is
#                                             image pull + weights + engine init (minutes)
#   startup_probe on /health                  vLLM answers 200 only after the KV cache is allocated

locals {
  served_model_name = coalesce(var.served_model_name, var.model)
  model_arg         = var.model_source == "gcs" ? "/models/${var.model}" : var.model
  vllm_args = concat([
    local.model_arg,
    "--served-model-name", local.served_model_name,
    "--port", "8000",
    "--max-model-len", tostring(var.max_model_len),
    "--gpu-memory-utilization", tostring(var.gpu_memory_utilization),
  ], var.extra_args)
}

resource "google_cloud_run_v2_service" "vllm" {
  project             = var.project_id
  name                = var.service_name
  location            = var.region
  description         = "vLLM OpenAI-compatible server on one NVIDIA L4 (vllm-serving-lab)"
  ingress             = "INGRESS_TRAFFIC_ALL" # reachable, but IAM-authenticated: see iam.tf
  deletion_protection = false
  labels              = var.labels

  template {
    service_account                  = google_service_account.vllm.email
    max_instance_request_concurrency = var.concurrency
    timeout                          = var.request_timeout
    labels                           = var.labels

    # Cheaper and needs less quota; the service is not spread across zones. # VERIFY: GPU quota type.
    gpu_zonal_redundancy_disabled = true

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    node_selector {
      accelerator = "nvidia-l4"
    }

    containers {
      name  = "vllm"
      image = var.image
      args  = local.vllm_args

      ports {
        container_port = 8000
      }

      resources {
        limits = {
          cpu              = var.cpu
          memory           = var.memory
          "nvidia.com/gpu" = "1"
        }
        cpu_idle          = false # GPU instances keep CPU allocated. # VERIFY: required for GPU services
        startup_cpu_boost = true
      }

      startup_probe {
        initial_delay_seconds = 0
        period_seconds        = 10
        timeout_seconds       = 5
        failure_threshold     = var.startup_failure_threshold
        http_get {
          path = "/health"
          port = 8000
        }
      }

      liveness_probe {
        period_seconds    = 30
        timeout_seconds   = 5
        failure_threshold = 3
        http_get {
          path = "/health"
          port = 8000
        }
      }

      env {
        name  = "HF_HUB_ENABLE_HF_TRANSFER" # parallel download when hf_transfer is in the image (verify)
        value = "1"
      }

      dynamic "env" {
        for_each = var.hf_token_secret_id == null ? [] : [var.hf_token_secret_id]
        content {
          name = "HF_TOKEN"
          value_source {
            secret_key_ref {
              secret  = env.value
              version = "latest"
            }
          }
        }
      }

      dynamic "volume_mounts" {
        for_each = var.model_source == "gcs" ? [1] : []
        content {
          name       = "weights"
          mount_path = "/models"
        }
      }
    }

    dynamic "volumes" {
      for_each = var.model_source == "gcs" ? [1] : []
      content {
        name = "weights"
        gcs {
          bucket    = var.weights_bucket
          read_only = true
        }
      }
    }
  }

  depends_on = [google_project_service.apis]

  lifecycle {
    precondition {
      condition     = var.model_source == "hf" || var.weights_bucket != null
      error_message = "model_source = \"gcs\" needs weights_bucket."
    }
  }
}
