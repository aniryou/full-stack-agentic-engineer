# deploy/gke — Kueue + DWS, ComputeClasses and GCS FUSE weights on the lab's GKE cluster

**What it does.** This folder holds five examples for the cluster that `deploy/gcp/terraform` creates. Each
example is one file, generated from `k8sgpu/gke.py`. You apply each one with `apply-examples.sh`:

| File | Shows | Needs |
|---|---|---|
| `00-smoke-l4.yaml` | Scale-from-zero of the Spot L4 pool, `nvidia-smi` from the GKE-installed driver | the Terraform |
| `10-kueue-gke.yaml` | Kueue on GKE: flavor `l4-spot` (plain quota) and flavor `l4-flex` behind the `dws-prov` AdmissionCheck. That AdmissionCheck uses a `ProvisioningRequest` of class `queued-provisioning.gke.io` | `install-addons.sh` |
| `20-dws-sample-job.yaml` | A 2-node gang. Kueue admits it only when DWS provisions both nodes. `provreq.kueue.x-k8s.io/maxRunDurationSeconds` sets the limit of the lease | `enable_flex_start_pool = true`, `install-addons.sh` |
| `30-computeclass-l4.yaml` | A custom ComputeClass: Spot, then on-demand, then flex-start, for the same L4 shape. GKE creates the node pools on demand | GKE with node-pool auto-creation for ComputeClasses (>= 1.33.3-gke.1136000, verify) |
| `40-serving-vllm-gcsfuse.yaml` | vLLM on one L4 through the ComputeClass. The FUSE CSI driver mounts the weights read-only from GCS. A startup probe has a size that suits the time to load the weights. The Deployment sets `maxSurge: 0` (a lab choice: with one replica, every rollout is an outage) | the weights copied to the bucket |
| `50-time-sharing-l4.yaml` | Four 1-GPU pods on one time-shared L4 (primer §9). The node advertises 4 `nvidia.com/gpu`. Every pod sees the same GPU. Nothing isolates the pods | `enable_time_sharing_pool = true` |

**Cost.** Each GPU example creates one or two `g2-standard-4` L4 nodes. The nodes stay while the pods of the
example run. A Spot node costs ~$0.25/h, an on-demand node costs ~$0.70/h, and flex-start has a discount (all verify).
The DWS Job ends after 5 minutes. The serving Deployment runs until you delete it.

**Clean up.** `deploy/gke/apply-examples.sh delete` drains the GPU nodes back to zero. Then run `terraform destroy`
(see `deploy/gcp/README.md`).

**The driver matters.** `vllm/vllm-openai:v0.30.0` is a CUDA 13.0.2 build. Its image config has
`CUDA_VERSION=13.0.2` and `VLLM_ENABLE_CUDA_COMPATIBILITY=0` (read from the registry on 2026-09-26). CUDA 13.0
needs an NVIDIA driver >= 580.65.06 (layer 02 primer §1.2).

The node pools that the ComputeClass creates do not inherit the `gpu_driver_version = LATEST` of the Terraform
pools. Thus each rung asks for `gpu.driverVersion: latest`. Without it, these node pools get the default branch
of GKE. It is possible that this branch is older (verify for your GKE version). If the driver and the image do
not match, the container exits at start-up with *"CUDA driver version is insufficient for CUDA runtime version"*.

**No Topology-Aware Scheduling on these pools.** The `l4-spot` and `l4-flex` flavors have no `topologyName`.
Google's own Kueue TAS examples use the `cloud.google.com/gce-topology-{block,subblock,host}` labels on A3, A4
and A4X node pools (GoogleCloudPlatform/cluster-toolkit `examples/gke-a3-*`, `gke-a4`, `gke-a4x`), but not on
G2/L4 (verify). Examine your nodes with `kubectl get nodes -L cloud.google.com/gce-topology-host`.
The TAS flavor of the kind lab carries over to A3/A4 pools with the real labels. On L4, it teaches only the
mechanism.

**Admission is not placement on `l4-spot`.** A plain-quota flavor admits a workload when the quota is free. Then
the autoscaler adds Spot nodes one at a time, and a stockout can leave part of a gang Running. The
`waitForPodsReady` setting of Kueue v0.19 (on by default: 30 min timeout, then evict and requeue with backoff)
corrects that partial gang. Adjust it in the `kueue-manager-config` ConfigMap in `kueue-system`. `l4-flex` needs it less,
because its ProvisioningRequest check admits only when every node exists.

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

The scripts print each command before they run it. With `DRY_RUN=1`, they print the commands and do not run
them. The scripts refuse a kubectl context that does not start with `gke_`, unless you set `KUBE_CONTEXT`.

## What to look at

* **Scale from zero** (`smoke`): first the pod is Pending with `FailedScheduling`. Then a `TriggeredScaleUp`
  event occurs. The node appears with the labels `cloud.google.com/gke-accelerator=nvidia-l4` and
  `cloud.google.com/gke-spot=true`, and with the GPU taint. Minutes pass before `Running`. That time is the cold
  start in the budget of notebook 04. A Spot stockout shows as `FailedScaleUp` (notebook 03).
* **DWS through Kueue** (`dws`): the Job stays suspended. The Workload gets `QuotaReserved=True`, and then the
  `dws-prov` check stays `Pending` while the ProvisioningRequest waits. When both nodes exist, the check changes
  to `Ready` and Kueue admits the Job. While the Job is suspended, there is no pod to examine, so read
  `kubectl get workloads,provisioningrequests -n ml`.
  `python3 -m k8sgpu pending --fixture gke-dws-waiting` shows how that stage reads (illustrative).
  When the pods exist, examine a live pod that is still Pending with
  `python3 -m k8sgpu pending --live "$(kubectl get pods -n ml -l job-name=dws-l4-gang -o jsonpath='{.items[0].metadata.name}')" -n ml`.
* **ComputeClass** (`computeclass`, `serving`): `kubectl get nodes -L cloud.google.com/compute-class,cloud.google.com/gke-spot`
  shows which rung provisioned the node. With `activeMigration`, GKE moves the pod back to Spot when Spot returns.
* **GCS FUSE** (`serving`): GKE injects a `gke-gcsfuse-sidecar` native sidecar into the pod. If there is no IAM
  binding, the pod shows `FailedMount` with a 403 (fixture `gcsfuse-permission-denied`). The requests of the sidecar
  (the `gke-gcsfuse/*-request` annotations) come from the same node bundle. The linter counts them.
* **Time-sharing** (`sharing`): read `kubectl get nodes -L cloud.google.com/gke-gpu-sharing-strategy,cloud.google.com/gke-max-shared-clients-per-gpu`
  and the allocatable `nvidia.com/gpu` of the node (4 for one L4). `kubectl logs -l job-name=shared-l4`
  prints the same GPU UUID four times.

## Clean up

```bash
deploy/gke/apply-examples.sh delete    # the GPU nodes drain back to zero within ~10 minutes
```

Then run `terraform destroy` (see `deploy/gcp/README.md`).

## Verify list

Repositories that Google owns confirm these items (2026-09-26):

- The queued-provisioning taint `cloud.google.com/gke-queued=true:NoSchedule` (GoogleCloudPlatform/cluster-toolkit).
- The `autoscaling.gke.io/provisioning-request` node label and the time-sharing node labels
  (GoogleCloudPlatform/cluster-autoscaler `labels/system_labels.go`).
- ComputeClass `gpu.driverVersion` (GoogleCloudPlatform/accelerated-platforms).

These items are not yet confirmed. Make sure of them:

- The `ResizeRequestName` detail that `podSetUpdates` uses.
- The ComputeClass field names (`kubectl explain computeclass.spec`).
- The driver branch that the `DEFAULT` of GKE installs.
- If G2/L4 nodes have GCE topology labels.
- The resource annotations of the GCS FUSE sidecar of GKE.
