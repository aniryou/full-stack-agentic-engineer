# ------------------------------------------------------------------------------------------
# Project and location
# ------------------------------------------------------------------------------------------

variable "project_id" {
  description = "Project to run the benchmark VM in. Needs a paid billing account (GPUs are not available on the Free Trial) and GPU quota."
  type        = string
}

variable "region" {
  description = "Region for the bucket and subnet. Must contain var.zone."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Zone for the VM. It must offer the GPU: gcloud compute accelerator-types list --filter=\"zone:us-central1-a\"."
  type        = string
  # VERIFY: L4 (and your Spot capacity) in this zone; try another zone on ZONE_RESOURCE_POOL_EXHAUSTED.
  default = "us-central1-a"
}

variable "name" {
  description = "Prefix for every resource name."
  type        = string
  default     = "gpubench"
}

variable "labels" {
  description = "Labels on every labelable resource (find the spend in billing reports by these)."
  type        = map(string)
  default     = { app = "gpubench", layer = "01-hardware" }
}

# ------------------------------------------------------------------------------------------
# The machine
# ------------------------------------------------------------------------------------------

variable "machine_type" {
  description = "g2-standard-4 = 1x L4 (bundled with the G2 shape). a2-highgpu-2g = 2x A100 40GB joined by NVLink (for the P2P notebook). For a T4 use an N1 shape plus accelerator_type."
  type        = string
  default     = "g2-standard-4"
}

variable "accelerator_type" {
  description = "Only for N1 machine types, which take a separate GPU (e.g. nvidia-tesla-t4). Leave null for G2/A2/A3, whose GPUs come with the shape."
  type        = string
  default     = null
}

variable "accelerator_count" {
  description = "How many accelerator_type GPUs to attach (N1 only)."
  type        = number
  default     = 1
}

variable "provisioning_model" {
  description = "SPOT (60-91% cheaper, can be preempted, capacity not guaranteed) or STANDARD (on-demand)."
  type        = string
  default     = "SPOT"

  validation {
    condition     = contains(["SPOT", "STANDARD"], var.provisioning_model)
    error_message = "provisioning_model must be SPOT or STANDARD."
  }
}

variable "max_run_duration_seconds" {
  description = "Safety net: Compute Engine stops or deletes the VM after this long even if the startup script hangs."
  type        = number
  default     = 3600

  validation {
    condition     = var.max_run_duration_seconds >= 600 && var.max_run_duration_seconds <= 86400
    error_message = "Use between 10 minutes and 24 hours."
  }
}

variable "termination_action" {
  description = "What happens at max_run_duration or on Spot preemption: DELETE (no disk left behind) or STOP."
  type        = string
  default     = "DELETE"

  validation {
    condition     = contains(["STOP", "DELETE"], var.termination_action)
    error_message = "termination_action must be STOP or DELETE."
  }
}

variable "image_project" {
  description = "Project that publishes the boot image (Deep Learning VM images)."
  type        = string
  default     = "deeplearning-platform-release"
}

variable "image_family" {
  description = "Deep Learning VM image family with CUDA and an NVIDIA driver. List current ones with: gcloud compute images list --project deeplearning-platform-release --no-standard-images --format='value(family)' | sort -u | grep cu12"
  type        = string
  # VERIFY: DLVM family names change with CUDA/driver releases; pick a current "common-cu12x-..." family.
  default = "common-cu128-ubuntu-2204-nvidia-570"
}

variable "boot_disk_size_gb" {
  description = "Boot disk size. The benchmark writes a synthetic checkpoint here (256 MB quick, 2 GB with --full)."
  type        = number
  default     = 100
}

variable "boot_disk_type" {
  description = "Boot disk type. Its throughput is what notebook 04's cold-read numbers measure on this VM."
  type        = string
  # VERIFY: supported disk types per machine series (newer shapes such as A3 Ultra/A4 may need hyperdisk-balanced).
  default = "pd-balanced"
}

variable "assign_public_ip" {
  description = "Give the VM an ephemeral external IP for egress (GitHub, PyPI). When false, Cloud NAT is created instead. Inbound is SSH from IAP only either way."
  type        = bool
  default     = true
}

variable "subnet_cidr" {
  description = "CIDR of the lab's subnet."
  type        = string
  default     = "10.42.0.0/24"
}

# ------------------------------------------------------------------------------------------
# What the VM runs
# ------------------------------------------------------------------------------------------

variable "repo_url" {
  description = "Git repository holding the lab (public: a plain shallow clone)."
  type        = string
  default     = "https://github.com/aniryou/full-stack-agentic-engineer.git"
}

variable "repo_ref" {
  description = "Branch or tag to check out."
  type        = string
  default     = "main"
}

variable "lab_path" {
  description = "Repository-relative path of the gpu-bench-lab directory."
  type        = string
  default     = "01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab"
}

variable "gpubench_args" {
  description = "Extra arguments for `python -m gpubench run`, e.g. \"--full\" (bigger sizes, ~10-20 min on an L4) or \"--suite gemm,stream\"."
  type        = string
  default     = ""
}

variable "torch_index_url" {
  description = "pip index for PyTorch when the image has none (\"\" = plain PyPI). The CUDA build must suit the image's driver."
  type        = string
  # VERIFY: cu128 wheels run on R525+ drivers (CUDA 12 minor-version compatibility); cu13x wheels need R580+.
  default = "https://download.pytorch.org/whl/cu128"
}

variable "power_off_when_done" {
  description = "Shut the VM down after uploading the report (the disk remains until terraform destroy, or until max_run_duration deletes it)."
  type        = bool
  default     = true
}

variable "results_retention_days" {
  description = "Objects in the results bucket are deleted after this many days."
  type        = number
  default     = 30
}

variable "enable_apis" {
  description = "Enable the Compute, Storage, IAM and Logging APIs (turn off if your organisation manages APIs centrally)."
  type        = bool
  default     = true
}
