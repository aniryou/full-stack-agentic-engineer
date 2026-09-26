# deploy/gcp — a GKE Sandbox (gVisor) cluster with no route to the internet

**What it does.** `terraform/` creates a zonal GKE Standard cluster with private nodes and Dataplane V2
(NetworkPolicy enforced), Workload Identity and Managed Prometheus; a one-node `e2-standard-2` system pool for
the proxy and the stand-in upstream; a **GKE Sandbox node pool** (`sandbox_config { type = "GVISOR" }`, COS with
containerd, Spot, autoscaling from 0 to 3, `podPidsLimit` 128, GKE metadata server); a least-privilege node
service account; an Artifact Registry repository for the sandbox image; and — only with `enable_nat = true` — a
Cloud Router and Cloud NAT. `mirror-image.sh` copies `python:3.12-slim` into that repository (private nodes
cannot reach Docker Hub). Then [`../gke/apply.sh`](../gke/) installs the sandbox platform.

**Cost.** Pay per use; prices change, look them up for your region and mark them (verify):
the system pool's `e2-standard-2` runs for as long as the cluster exists (roughly $0.07/h on demand in
us-central1, verify); the GKE management fee is covered by the free tier for one zonal cluster per billing
account (verify); the gVisor pool costs nothing at zero nodes and a Spot `e2-standard-2` per node while
executions run (Spot is 60-91 % off on-demand, verify); GKE Sandbox has no surcharge of its own (verify);
Artifact Registry storage for one ~50 MB image and Managed Prometheus ingestion are cents per month at lab volume
(verify). Cloud NAT, if you enable it, bills per hour and per GB. See [`COMPUTE.md`](../../../../../COMPUTE.md).

**Clean up.** `terraform -chdir=deploy/gcp/terraform destroy` removes everything this directory created
(`deletion_protection` is off by default). The images in Artifact Registry go with the repository.

## Run it

```bash
cd deploy/gcp/terraform
cp terraform.tfvars.example terraform.tfvars      # set project_id (and authorized_networks to your IP/32)
terraform init && terraform plan                  # read the plan: 2 pools, no NAT, GVISOR
terraform apply                                   # ~10 minutes
cd ../../.. && deploy/gcp/mirror-image.sh         # python:3.12-slim -> Artifact Registry
deploy/gke/apply.sh                               # the sandbox platform + one execution under gVisor
python3 -m sandboxlab gke-review                  # the design-review checklist, read from these .tf files
```

`gcloud` equivalent of the cluster and pool: `python3 -c "from sandboxlab import gke; print(gke.GCLOUD_EQUIVALENT)"`.

## Design notes

* **`type = "GVISOR"`** — the provider validates the value case-sensitively; `gcloud`'s flag is lowercase
  (`--sandbox type=gvisor`). GKE Sandbox needs a second node pool (the first cannot be sandboxed), and GKE adds
  the node label and taint `sandbox.gke.io/runtime=gvisor` itself and creates RuntimeClass `gvisor` (verify),
  so no taint is declared here and pods need only `runtimeClassName: gvisor`.
* **No internet egress by default** — private nodes and no NAT: the only destinations are Google APIs over
  Private Google Access (verify that it covers `*.pkg.dev` for your setup). Those APIs are still a way out (an
  attacker's bucket is a Google API), so the sandbox namespace's default-deny NetworkPolicy (or VPC Service
  Controls) is what closes them. With `enable_nat = true` the whole subnet gets a route out; from then on only
  NetworkPolicy (Dataplane V2) keeps sandboxes off the internet — gVisor does not: its default
  `--network=sandbox` gives the workload a full userspace network stack.
* **Metadata server** — a NetworkPolicy cannot block the node-local metadata path (verify); the defence is
  `GKE_METADATA` mode, `automountServiceAccountToken: false` and a KSA with no IAM binding.
* **Agent Sandbox add-on** — `enable_agent_sandbox_addon` turns on GKE's managed kubernetes-sigs/agent-sandbox
  (verify availability); `../gke/apply.sh WITH_AGENT_SANDBOX=1` installs the upstream controller instead.
