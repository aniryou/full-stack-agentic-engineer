# deploy/gke — Kueue + DWS, ComputeClasses and GCS FUSE weights on the lab's GKE cluster

**What it does.** Five examples for the cluster that `deploy/gcp/terraform` creates, each one
file (generated from `k8sgpu/gke.py`), each applied with `apply-examples.sh`:

| File | Shows | Needs |
|---|---|---|
| `00-smoke-l4.yaml` | scale-from-zero of the Spot L4 pool; `nvidia-smi` from the GKE-installed driver | the Terraform |
| `10-kueue-gke.yaml` | Kueue on GKE: flavor `l4-spot` (plain quota) and flavor `l4-flex` behind the `dws-prov` AdmissionCheck — a `ProvisioningRequest` of class `queued-provisioning.gke.io` | `install-addons.sh` |
| `20-dws-sample-job.yaml` | a 2-node gang admitted only when DWS has provisioned both nodes; `provreq.kueue.x-k8s.io/maxRunDurationSeconds` bounds the lease | `enable_flex_start_pool = true`, `install-addons.sh` |
| `30-computeclass-l4.yaml` | a custom ComputeClass: Spot → on-demand → flex-start for the same L4 shape, node pools created on demand | GKE with node-pool auto-creation for ComputeClasses (>= 1.33.3-gke.1136000, verify) |
| `40-serving-vllm-gcsfuse.yaml` | vLLM on one L4 through the ComputeClass; weights mounted read-only from GCS by the FUSE CSI driver; a startup probe sized for the load; `maxSurge: 0` (a lab choice: one replica means every rollout is an outage) | the weights copied to the bucket |
| `50-time-sharing-l4.yaml` | four 1-GPU pods on one time-shared L4 (primer §9): the node advertises 4 `nvidia.com/gpu`, every pod sees the same GPU, nothing isolates them | `enable_time_sharing_pool = true` |

**The driver matters.** `vllm/vllm-openai:v0.30.0` is a CUDA 13.0.2 build (its image config:
`CUDA_VERSION=13.0.2`, `VLLM_ENABLE_CUDA_COMPATIBILITY=0`; read from the registry on 2026-09-26),
and CUDA 13.0 needs an NVIDIA driver >= 580.65.06 (layer 02 primer §1.2). The node pools the
ComputeClass creates do not inherit the Terraform pools' `gpu_driver_version = LATEST`, so each
rung asks for `gpu.driverVersion: latest`; without it they get GKE's default branch, which may be
older (verify for your GKE version). Symptom of a mismatch: the container exits at start-up with
*"CUDA driver version is insufficient for CUDA runtime version"*.

**No Topology-Aware Scheduling on these pools.** The `l4-spot` and `l4-flex` flavors have no
`topologyName`. Google's own Kueue TAS examples use the `cloud.google.com/gce-topology-{block,subblock,host}`
labels on A3, A4 and A4X node pools (GoogleCloudPlatform/cluster-toolkit `examples/gke-a3-*`,
`gke-a4`, `gke-a4x`), not on G2/L4; check yours with
`kubectl get nodes -L cloud.google.com/gce-topology-host` (verify). The kind lab's TAS flavor
carries over to A3/A4 pools with the real labels; on L4 it teaches the mechanism only.

**Admission is not placement on `l4-spot`.** A plain-quota flavor admits when the quota is free;
the autoscaler then adds Spot nodes one at a time, and a stockout can leave part of a gang
Running. Kueue v0.19's `waitForPodsReady` (on by default: 30 min timeout, then evict and requeue
with backoff) cleans that up; tune it in the `kueue-manager-config` ConfigMap in `kueue-system`.
`l4-flex` does not need it as much: its ProvisioningRequest check admits only when every node exists.

**Cost.** Each GPU example creates one or two `g2-standard-4` L4 nodes for as long as its pods
run: ~$0.25/h per Spot node, ~$0.70/h on-demand, flex-start discounted (all verify). The DWS
Job ends after 5 minutes; the serving Deployment runs until you delete it.

## Run it

```bash
$(terraform -chdir=deploy/gcp/terraform output -raw get_credentials)
deploy/gke/apply-examples.sh smoke            # watch: kubectl get pods -w ; kubectl get events -w
deploy/gke/install-addons.sh                  # JobSet + Kueue (same pinned versions as kind) + 10-kueue-gke.yaml
deploy/gke/apply-examples.sh dws              # watch: kubectl get workloads,provisioningrequests -n ml -w
deploy/gke/apply-examples.sh computeclass
deploy/gke/apply-examples.sh sharing          # needs enable_time_sharing_pool = true

# weights for the serving example (any small model works; this one is ~3 GB):
python3 -m pip install -U huggingface_hub          # provides the `hf` CLI
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir qwen2.5-1.5b-instruct
gcloud storage cp -r qwen2.5-1.5b-instruct "gs://$(terraform -chdir=deploy/gcp/terraform output -raw weights_bucket)/"
WEIGHTS_BUCKET=$(terraform -chdir=deploy/gcp/terraform output -raw weights_bucket) deploy/gke/apply-examples.sh serving
kubectl -n serving port-forward svc/vllm-l4 8000:8000 &
curl -s localhost:8000/v1/models
```

Every command is printed first; `DRY_RUN=1` prints without running. The scripts refuse a
kubectl context that does not start with `gke_` unless `KUBE_CONTEXT` is set.

## What to look at

* **Scale from zero** (`smoke`): the pod is Pending with `FailedScheduling`, then a
  `TriggeredScaleUp` event; the node appears with labels `cloud.google.com/gke-accelerator=nvidia-l4`,
  `cloud.google.com/gke-spot=true` and the GPU taint; minutes pass before `Running` — that is
  the cold start notebook 04 budgets. A Spot stockout shows as `FailedScaleUp` (notebook 03).
* **DWS through Kueue** (`dws`): the Job stays suspended; the Workload gets `QuotaReserved=True`,
  then the `dws-prov` check stays `Pending` while the ProvisioningRequest waits; when both nodes
  exist it turns `Ready` and the Job is admitted. `python3 -m k8sgpu pending` explains each stage.
* **ComputeClass** (`computeclass`, `serving`): `kubectl get nodes -L cloud.google.com/compute-class,cloud.google.com/gke-spot`
  shows which rung provisioned; with `activeMigration` GKE moves the pod back to Spot when it returns.
* **GCS FUSE** (`serving`): a `gke-gcsfuse-sidecar` native sidecar is injected into the pod; a
  missing IAM binding shows as `FailedMount` with a 403 (fixture `gcsfuse-permission-denied`).
  Its requests (the `gke-gcsfuse/*-request` annotations) come out of the same node bundle; the
  linter counts them.
* **Time-sharing** (`sharing`): `kubectl get nodes -L cloud.google.com/gke-gpu-sharing-strategy,cloud.google.com/gke-max-shared-clients-per-gpu`
  and the node's allocatable `nvidia.com/gpu` (4 for one L4); `kubectl logs -l job-name=shared-l4`
  prints the same GPU UUID four times.

## Clean up

```bash
deploy/gke/apply-examples.sh delete    # the GPU nodes drain back to zero within ~10 minutes
```

Then `terraform destroy` (see `deploy/gcp/README.md`).

## Verify list

Corroborated from Google-owned repositories (2026-09-26): the queued-provisioning taint
`cloud.google.com/gke-queued=true:NoSchedule` (GoogleCloudPlatform/cluster-toolkit), the
`autoscaling.gke.io/provisioning-request` node label and the time-sharing node labels
(GoogleCloudPlatform/cluster-autoscaler `labels/system_labels.go`), ComputeClass
`gpu.driverVersion` (GoogleCloudPlatform/accelerated-platforms). Still verify: the
`ResizeRequestName` detail used by `podSetUpdates`; ComputeClass field names
(`kubectl explain computeclass.spec`); the driver branch GKE's `DEFAULT` installs; whether G2/L4
nodes carry GCE topology labels; GKE's GCS FUSE sidecar resource annotations.
