# gpubench on GCP (T3): one Spot L4 VM, report to a bucket, auto-stop

**What it does.** Terraform creates one Spot `g2-standard-4` VM (1× L4). The VM runs the whole suite when it boots.
Then it uploads the JSON and Markdown report to a private bucket and powers itself off. `bench-on-gcp.sh` does the
whole cycle in one script: apply, wait, download, destroy.

**Cost.** The VM costs approximately $0.07–0.28/hr on Spot (us-central1, September 2026, verify). The disk adds some
cents. A short run takes much less than one hour. The details are in
[Cost](#cost-approximate-us-central1-september-2026--verify).

**Clean up.** `bench-on-gcp.sh` destroys all the resources for you. If you work by hand, run `terraform destroy`. A
VM that powered itself off stops the GPU billing. But the stopped VM and its boot disk stay until you destroy them.

The configuration in [`terraform/`](terraform/) has one file for each concern. It creates these resources:

| File | Resources |
|---|---|
| `main.tf` | a random suffix, shared labels, the Compute/Storage/IAM/Logging APIs |
| `network.tf` | A VPC and a subnet. A firewall that permits SSH **only** from Identity-Aware Proxy. Cloud NAT when `assign_public_ip = false`. |
| `storage.tf` | A private bucket for the results. It has uniform access and prevents public access. It deletes objects after 30 days. |
| `iam.tf` | A service account that can write objects to that bucket and write logs. It can do nothing else. |
| `compute.tf` | One `g2-standard-4` (1× L4) **Spot** VM from a Deep Learning VM image. `max_run_duration` = 1 h. The termination action is DELETE. |
| `startup.sh` | It runs when the VM boots. It waits for the driver and records `nvidia-smi`, the topology and the inventory. Then it installs the lab and runs `python -m gpubench run --backend torch`. It uploads to `gs://<bucket>/results/<vm>-<time>/` and powers off. If *any* failure occurs, it writes `FAILED.txt`, uploads the files that exist and powers off. |

The VM measures the L4 *and* its own boot disk. For the L4, it measures tensor-core GEMMs (FP8 included), GDDR6
bandwidth and PCIe Gen4 copies between host and device. For the boot disk, it measures the cold reads of notebook
04. The result is a clean cloud baseline to put beside your laptop and Colab runs.

## Before you start

* A project with a **paid** billing account. You cannot use GPUs on the Free Trial. An upgrade keeps the credits.
* GPU quota: `GPUS_ALL_REGIONS` ≥ 1 and the regional L4 quota (`NVIDIA_L4_GPUS`). New projects often have 0. Request
  the quota in *IAM & Admin*, then *Quotas*. T4 and L4 requests usually get a fast approval. A100/H100 are hard to
  get on new or individual accounts.
* Terraform ≥ 1.9 and the Google Cloud CLI. Authenticate with `gcloud auth application-default login`.

## Cost (approximate, us-central1, September 2026 — verify)

`g2-standard-4` costs about $0.70/hr on demand. Spot usually costs 60–91% less, thus approximately $0.07–0.28/hr.
There is also a 100 GB pd-balanced boot disk (cents per day) and an ephemeral external IP. A short run takes much
less than one hour, and then the VM powers itself off. The GPU billing stops at that time. But the **stopped VM and
its boot disk stay until `terraform destroy`**. (`bench-on-gcp.sh` destroys them for you. If you work by hand, run
the command yourself.)

`max_run_duration` is the safety net for a *hung* run. It deletes a VM after one hour in the RUNNING state. It
counts only the time in the RUNNING state. Thus it does not clean up a VM that already stopped itself (verify).
`a2-highgpu-2g` (2× A100 40GB with NVLink, for the P2P notebook) costs about $7/hr on demand. Use Spot if the
capacity permits it, and keep the duration short.

## Run it

One command does the full cycle: apply, wait for the report, download to `results/gcp/`, destroy. If something
fails after the apply, the script copies the files that reached the bucket. Then it destroys everything all the
same. No report within `TIMEOUT_MIN` (default 45) is also a failure. `KEEP=1` keeps the resources instead:

```bash
PROJECT=my-gpu-lab deploy/gcp/bench-on-gcp.sh
PROJECT=my-gpu-lab TF_VARS="-var machine_type=a2-highgpu-2g" deploy/gcp/bench-on-gcp.sh   # NVLink P2P
DRY_RUN=1 PROJECT=x deploy/gcp/bench-on-gcp.sh     # print the steps only
```

Or do the steps by hand:

```bash
cd deploy/gcp/terraform
cp terraform.tfvars.example terraform.tfvars      # set project_id; pick the machine
terraform init && terraform apply
$(terraform output -raw watch_progress)           # the startup script prints [gpubench] steps to the serial console
$(terraform output -raw fetch_results)            # once the report is in the bucket
python3 -m gpubench show results-gcp/results/*/gpubench-*.json
terraform destroy                                 # removes the VM, network, service account and the bucket
```

## Switching machines

| Want | Set |
|---|---|
| the default: 1× L4, Ada, FP8 GEMMs | `machine_type = "g2-standard-4"` |
| NVLink P2P between two GPUs (notebook 03) | `machine_type = "a2-highgpu-2g"` (2× A100 40GB) |
| a T4 (Turing: fp16 only, PCIe Gen3) | `machine_type = "n1-standard-8"`, `accelerator_type = "nvidia-tesla-t4"` |
| on-demand instead of Spot | `provisioning_model = "STANDARD"` |
| larger sizes | `gpubench_args = "--full"` (tens of minutes, thus increase `max_run_duration_seconds`) |

You get H100 and newer shapes mostly through reservations, DWS flex-start or calendar mode. `a3-highgpu-8g` is
available on demand at approximately $88/hr. The smaller A3 shapes are available only through Spot or flex-start.
A3 Ultra (H200) and A4 (B200) are mostly for reservations (verify). Several of these shapes need Hyperdisk boot
disks. Layer 03 covers how to get that capacity.

## Troubleshooting

* `ZONE_RESOURCE_POOL_EXHAUSTED`, or the Spot VM disappears: there is no capacity at this time. Try another zone
  that has the GPU, try again later, or use `STANDARD`.
* `Quota 'NVIDIA_L4_GPUS' exceeded`: request quota (see "Before you start").
* The report contains `FAILED.txt`: the driver did not start, or the suite failed. The serial console log
  (`watch_progress`) shows which step.
* No report, and the VM is gone: the cause is Spot preemption or `max_run_duration`. Run again.
* You ran Terraform by hand and the run finished: the VM is *stopped*, not deleted. Run `terraform destroy`.

## VERIFY before relying on it

* `image_family` default `common-cu128-ubuntu-2204-nvidia-570`: the names of Deep Learning VM families change with
  CUDA and driver releases. After some time, an older family gets no more images. Make sure that this family is
  still current before you rely on it. List the current families with
  `gcloud compute images list --project deeplearning-platform-release --no-standard-images --format='value(family)' | sort -u`.
  A family with an `nvidia-580` driver (for example `common-cu129-ubuntu-2404-nvidia-580`, if it is in the list,
  verify) also runs CUDA 13 PyTorch wheels. These wheels need R580+.
* The metadata `install-nvidia-driver = True` installs the driver at the first boot. It does this for DLVM images
  that do not come with a preinstalled driver.
* `boot_disk_type = pd-balanced` is correct for G2/A2/N1. Newer shapes (A3 Ultra, A4, ...) can need
  `hyperdisk-balanced`.
* `torch_index_url` (default: CUDA 12.8 wheels) must agree with the driver of the image. CUDA 12.x needs R525+
  through minor-version compatibility. Blackwell needs R570+. The newest PyTorch with cu128 wheels is 2.11 (Docker
  Hub has no later `cuda12.8` tag). Thus this VM runs an older PyTorch than the Docker path.
* `max_run_duration` counts only the time in the RUNNING state. It does not delete a VM that powered itself off.
* L4 availability in `us-central1-a`, Spot pricing and discounts.

The topic primer, [`../../../PRIMER.md`](../../../PRIMER.md), explains the concepts behind the numbers. Its §10
"Getting hardware" covers the GCP families and how to get them.
