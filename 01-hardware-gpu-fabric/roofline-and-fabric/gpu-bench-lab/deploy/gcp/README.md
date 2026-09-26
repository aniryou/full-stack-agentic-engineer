# gpubench on GCP (T3): one Spot L4 VM, report to a bucket, auto-stop

What it creates ([`terraform/`](terraform/), split by concern):

| File | Resources |
|---|---|
| `main.tf` | a random suffix, shared labels, the Compute/Storage/IAM/Logging APIs |
| `network.tf` | a VPC + subnet; firewall allowing SSH **only** from Identity-Aware Proxy; Cloud NAT when `assign_public_ip = false` |
| `storage.tf` | a private results bucket (uniform access, public access prevented, objects deleted after 30 days) |
| `iam.tf` | a service account that can write objects to that bucket and write logs — nothing else |
| `compute.tf` | one `g2-standard-4` (1× L4) **Spot** VM from a Deep Learning VM image, `max_run_duration` = 1 h, termination action DELETE |
| `startup.sh` | runs on boot: waits for the driver, records `nvidia-smi` / topology / inventory, installs the lab, runs `python -m gpubench run --backend torch`, uploads to `gs://<bucket>/results/<vm>-<time>/`, powers off — and on *any* failure writes `FAILED.txt`, uploads what exists and powers off |

The VM measures the L4 (tensor-core GEMMs including FP8, GDDR6 bandwidth, PCIe Gen4 host↔device
copies) *and* its own boot disk (notebook 04's cold reads) — a clean cloud baseline to set beside your
laptop and Colab runs.

## Before you start

* A project with a **paid** billing account — GPUs cannot be used on the Free Trial (upgrading keeps
  the credits).
* GPU quota: `GPUS_ALL_REGIONS` ≥ 1 and the regional L4 quota (`NVIDIA_L4_GPUS`), often 0 on new
  projects — request it in *IAM & Admin → Quotas*. T4 and L4 requests are usually approved quickly;
  A100/H100 are hard to get on new or individual accounts.
* Terraform ≥ 1.9 and the Google Cloud CLI, authenticated: `gcloud auth application-default login`.

## Cost (approximate, us-central1, September 2026 — verify)

`g2-standard-4` is about $0.70/hr on demand; Spot is typically 60–91% cheaper, so roughly
$0.07–0.28/hr, plus a 100 GB pd-balanced boot disk (cents per day) and an ephemeral external IP. A
quick run is done in well under an hour, and the VM powers itself off — GPU billing stops, but the
**stopped VM and its boot disk stay until `terraform destroy`** (`bench-on-gcp.sh` destroys for you;
by hand, run it yourself). `max_run_duration` is the net for a *hung* run: it deletes a VM that has
been running for an hour; it counts running time only, so it does not clean up a VM that already
stopped itself (verify). `a2-highgpu-2g` (2× A100 40GB with NVLink, for the P2P notebook) is about
$7/hr on demand — use Spot if capacity allows and keep the duration short.

## Run it

One command (apply → wait for the report → download to `results/gcp/` → destroy). If anything
fails after the apply — including no report within `TIMEOUT_MIN` (default 45) — it copies whatever
reached the bucket and destroys everything anyway; `KEEP=1` keeps the resources instead:

```bash
PROJECT=my-gpu-lab deploy/gcp/bench-on-gcp.sh
PROJECT=my-gpu-lab TF_VARS="-var machine_type=a2-highgpu-2g" deploy/gcp/bench-on-gcp.sh   # NVLink P2P
DRY_RUN=1 PROJECT=x deploy/gcp/bench-on-gcp.sh     # print the steps only
```

Or by hand:

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
| bigger sizes | `gpubench_args = "--full"` (tens of minutes; raise `max_run_duration_seconds`) |

H100 and newer shapes are mostly obtained through reservations, DWS flex-start or calendar mode:
`a3-highgpu-8g` is on demand at roughly $88/hr, the smaller A3 shapes only via Spot or flex-start,
and A3 Ultra (H200) / A4 (B200) are reservation-oriented (verify); several need Hyperdisk boot disks.
Getting that capacity is layer 03's territory.

## Troubleshooting

* `ZONE_RESOURCE_POOL_EXHAUSTED` or the Spot VM disappears — no capacity right now: try another zone
  that offers the GPU, later, or `STANDARD`.
* `Quota 'NVIDIA_L4_GPUS' exceeded` — request quota (above).
* The report contains `FAILED.txt` — the driver did not come up or the suite failed; the serial
  console log (`watch_progress`) shows which step.
* No report and the VM is gone — Spot preemption or `max_run_duration`; run again.
* You ran Terraform by hand and the run finished — the VM is *stopped*, not deleted: `terraform destroy`.

## VERIFY before relying on it

* `image_family` default `common-cu128-ubuntu-2204-nvidia-570`: Deep Learning VM family names change
  with CUDA and driver releases, and older families stop receiving images (check that this one is
  still current before relying on it) — list current ones with
  `gcloud compute images list --project deeplearning-platform-release --no-standard-images --format='value(family)' | sort -u`.
  A family with an `nvidia-580` driver (for example `common-cu129-ubuntu-2404-nvidia-580`, if listed —
  verify) also runs CUDA 13 PyTorch wheels, which need R580+.
* `install-nvidia-driver = True` metadata installs the driver on first boot for DLVM images that do
  not ship one preinstalled.
* `boot_disk_type = pd-balanced` suits G2/A2/N1; newer shapes (A3 Ultra, A4, ...) may require `hyperdisk-balanced`.
* `torch_index_url` (default CUDA 12.8 wheels) must suit the image's driver (R525+ for CUDA 12.x
  through minor-version compatibility; R570+ for Blackwell). The newest PyTorch with cu128 wheels is
  2.11 (Docker Hub has no later `cuda12.8` tag), so this VM runs an older PyTorch than the Docker path.
* `max_run_duration` counts only time in the RUNNING state: a VM that powered itself off is not deleted by it.
* L4 availability in `us-central1-a`, Spot pricing and discounts.

The concepts behind the numbers are in the topic primer, [`../../../PRIMER.md`](../../../PRIMER.md)
(§10 "Getting hardware" covers GCP families and obtainability).
