# One GPU VM on Spot, from a Deep Learning VM image. Its startup script (startup.sh) installs the
# lab, runs `python -m gpubench run --backend torch`, uploads the JSON + Markdown report to the
# bucket and powers off. max_run_duration is the safety net if anything hangs.

resource "google_compute_instance" "bench" {
  name                      = local.name
  machine_type              = var.machine_type
  zone                      = var.zone
  labels                    = local.labels
  tags                      = [var.name]
  allow_stopping_for_update = true

  boot_disk {
    initialize_params {
      image  = "projects/${var.image_project}/global/images/family/${var.image_family}"
      size   = var.boot_disk_size_gb
      type   = var.boot_disk_type
      labels = local.labels
    }
  }

  # G2/A2/A3 shapes come with their GPUs; N1 takes them separately.
  dynamic "guest_accelerator" {
    for_each = var.accelerator_type == null ? [] : [var.accelerator_type]
    content {
      type  = guest_accelerator.value
      count = var.accelerator_count
    }
  }

  scheduling {
    provisioning_model          = var.provisioning_model
    preemptible                 = var.provisioning_model == "SPOT"
    automatic_restart           = false
    on_host_maintenance         = "TERMINATE" # GPU VMs cannot live-migrate
    instance_termination_action = var.termination_action

    max_run_duration {
      seconds = var.max_run_duration_seconds
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.bench.id

    dynamic "access_config" {
      for_each = var.assign_public_ip ? [1] : []
      content {}
    }
  }

  service_account {
    email  = google_service_account.bench.email
    scopes = ["cloud-platform"] # access is limited by the IAM roles in iam.tf, not by scopes
  }

  metadata = {
    # VERIFY: Deep Learning VM images install the NVIDIA driver on first boot when this is set.
    "install-nvidia-driver" = "True"
    "enable-oslogin"        = "TRUE"
    "startup-script"        = file("${path.module}/startup.sh")
    "gpubench-bucket"       = google_storage_bucket.results.name
    "gpubench-repo"         = var.repo_url
    "gpubench-ref"          = var.repo_ref
    "gpubench-lab-path"     = var.lab_path
    "gpubench-args"         = var.gpubench_args
    "gpubench-poweroff"     = var.power_off_when_done ? "true" : "false"
    "gpubench-torch-index"  = var.torch_index_url
  }

  depends_on = [
    google_storage_bucket_iam_member.results_writer,
    google_project_iam_member.log_writer,
    google_compute_router_nat.nat,
  ]
}
