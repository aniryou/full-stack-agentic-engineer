# deploy/gke — Kueue + DWS, ComputeClasses and GCS FUSE weights on the lab's GKE cluster

**What it does.** Four examples for the cluster that `deploy/gcp/terraform` creates, each one
file (generated from `k8sgpu/gke.py`), each applied with `apply-examples.sh`:

| File | Shows | Needs |
|---|---|---|
| `00-smoke-l4.yaml` | scale-from-zero of the Spot L4 pool; `nvidia-smi` from the GKE-installed driver | the Terraform |
| `10-kueue-gke.yaml` | Kueue on GKE: flavor `l4-spot` (plain quota) and flavor `l4-flex` behind the `dws-prov` AdmissionCheck — a `ProvisioningRequest` of class `queued-provisioning.gke.io` | `install-addons.sh` |
| `20-dws-sample-job.yaml` | a 2-node gang admitted only when DWS has provisioned both nodes; `provreq.kueue.x-k8s.io/maxRunDurationSeconds` bounds the lease | `enable_flex_start_pool = true`, `install-addons.sh` |
| `30-computeclass-l4.yaml` | a custom ComputeClass: Spot → on-demand → flex-start for the same L4 shape, node pools created on demand | GKE with node-pool auto-creation for ComputeClasses (>= 1.33.3-gke.1136000, verify) |
| `40-serving-vllm-gcsfuse.yaml` | vLLM on one L4 through the ComputeClass; weights mounted read-only from GCS by the FUSE CSI driver; a startup probe sized for the load; `maxSurge: 0` | the weights copied to the bucket |

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

## Clean up

```bash
deploy/gke/apply-examples.sh delete    # the GPU nodes drain back to zero within ~10 minutes
```

Then `terraform destroy` (see `deploy/gcp/README.md`).

## Verify list

The queued-provisioning taint key (`cloud.google.com/gke-queued`) the `l4-flex` flavor tolerates;
the `autoscaling.gke.io/provisioning-request` node label and `ResizeRequestName` detail used by
`podSetUpdates`; ComputeClass field names (`kubectl explain computeclass.spec`); that the
vLLM image's CUDA build matches the node driver; GKE's GCS FUSE sidecar resource annotations.
