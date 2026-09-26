# ------------------------------------------------------------------------------------------
# Where
# ------------------------------------------------------------------------------------------
variable "project_id" {
  description = "Project that hosts the service. Needs a paid billing account (GPUs are not available on the Free Trial) and Cloud Run L4 quota in var.region."
  type        = string
}

variable "region" {
  description = "Cloud Run region with L4 GPUs. # VERIFY: the current list of GPU regions in the Cloud Run GPU docs."
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Cloud Run service name."
  type        = string
  default     = "vllm-l4"
}

variable "labels" {
  description = "Labels on every labelable resource (cost attribution)."
  type        = map(string)
  default     = { app = "vllm-serving-lab", layer = "04-inference-engine" }
}

# ------------------------------------------------------------------------------------------
# What runs: the engine image, the model and the engine flags
# ------------------------------------------------------------------------------------------
variable "image" {
  description = "vLLM OpenAI-compatible server image (ENTRYPOINT is `vllm serve`). Pin a version: v0.30.0 is on Docker Hub (checked 2026-09-26)."
  type        = string
  default     = "vllm/vllm-openai:v0.30.0"
}

variable "model" {
  description = "Hugging Face model id (model_source = \"hf\"), or the path inside the weights bucket (model_source = \"gcs\")."
  type        = string
  default     = "Qwen/Qwen2.5-1.5B-Instruct"
}

variable "served_model_name" {
  description = "Model name clients send in the OpenAI `model` field. Defaults to var.model."
  type        = string
  default     = null
}

variable "model_source" {
  description = "\"hf\": download from Hugging Face at start-up (HF token optional, from Secret Manager). \"gcs\": read weights from a Cloud Storage bucket mounted at /models (faster, repeatable cold starts)."
  type        = string
  default     = "hf"

  validation {
    condition     = contains(["hf", "gcs"], var.model_source)
    error_message = "model_source must be \"hf\" or \"gcs\"."
  }
}

variable "hf_token_secret_id" {
  description = "Secret Manager secret id holding a Hugging Face token (for gated models such as Llama). Create it yourself (`gcloud secrets create hf-token --data-file=-`) so the token never enters Terraform state. null = no token."
  type        = string
  default     = null
}

variable "weights_bucket" {
  description = "Bucket with model weights (model_source = \"gcs\"). Created here when create_weights_bucket = true."
  type        = string
  default     = null

  # Checked at plan time, before any resource (the bucket IAM grant included) is evaluated.
  validation {
    condition     = var.model_source == "hf" || var.weights_bucket != null
    error_message = "model_source = \"gcs\" needs weights_bucket (an existing bucket, or one to create with create_weights_bucket = true)."
  }
}

variable "create_weights_bucket" {
  description = "Create var.weights_bucket (regional, same region as the service). Upload weights with `gcloud storage cp -r`."
  type        = bool
  default     = false
}

variable "max_model_len" {
  description = "--max-model-len. Size it first: `python -m servelab size --model <config> --gpu L4 --max-model-len N`."
  type        = number
  default     = 8192
}

variable "gpu_memory_utilization" {
  description = "--gpu-memory-utilization: the fraction of the L4's memory vLLM may use (weights + activations + KV blocks)."
  type        = number
  default     = 0.92
}

variable "extra_args" {
  description = "More `vllm serve` flags, e.g. [\"--max-num-seqs\", \"64\", \"--kv-cache-dtype\", \"fp8\", \"--api-key\", \"...\"]."
  type        = list(string)
  default     = []
}

# ------------------------------------------------------------------------------------------
# Instance shape and scaling (the cost knobs)
# ------------------------------------------------------------------------------------------
variable "cpu" {
  description = "vCPUs per instance. An L4 instance needs at least 4 vCPU and 16 GiB; 8 vCPU / 32 GiB leaves room to download and load weights."
  type        = string
  default     = "8"
}

variable "memory" {
  description = "Memory per instance. Files downloaded into the container (the HF cache) live in this in-memory filesystem."
  type        = string
  default     = "32Gi"
}

variable "min_instances" {
  description = "0 = scale to zero (pay nothing while idle, pay a cold start on the next request). 1 = always warm (pay every second)."
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Upper bound on instances = on GPUs. Your L4 quota in this region caps it too."
  type        = number
  default     = 1
}

variable "concurrency" {
  description = "Requests Cloud Run sends to one instance at once (max_instance_request_concurrency). Match it to the batch the engine serves within your SLO (notebook 06); beyond it, requests queue at Cloud Run or start a new instance."
  type        = number
  default     = 32
}

variable "request_timeout" {
  description = "Per-request timeout: long generations stream for a while."
  type        = string
  default     = "600s"
}

variable "startup_failure_threshold" {
  description = "Startup probe failures allowed at 10 s intervals before the instance is restarted: 60 = 10 minutes for pull + weights + engine init. # VERIFY: Cloud Run startup-probe limits."
  type        = number
  default     = 60
}

# ------------------------------------------------------------------------------------------
# Who may call it
# ------------------------------------------------------------------------------------------
variable "invoker_members" {
  description = "Principals granted roles/run.invoker, e.g. [\"user:you@example.com\"]. Callers send `Authorization: Bearer $(gcloud auth print-identity-token)`. Keep the service private: an open LLM endpoint is a cost and abuse risk."
  type        = list(string)
  default     = []
}
