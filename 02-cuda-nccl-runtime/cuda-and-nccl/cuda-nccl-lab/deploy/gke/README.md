# deploy/gke — GPU workloads on GKE (T3)

**What it does:** it runs the GPU checks of the lab on the cluster from `../gcp/terraform`, or on any
GKE cluster with L4 nodes. The checks are:

- a smoke test that records how a pod sees its GPU
- a CUDA sample
- nccl-tests across two L4s
- GPU time-sharing
- a MIG slice
- DCGM alert rules for Google Managed Prometheus

**Cost:** each Job starts a Spot GPU node from zero. It releases the node when the pod finishes. The
cluster autoscaler removes a node that is not necessary after approximately ten minutes (verify for
your version).

An L4 Spot node for 15 minutes costs some cents. The 2 x L4 node for nccl-tests costs approximately
two times that (verify current Spot prices). Nothing in this folder runs with no time limit, except
the workload of `04-time-sharing.yaml`. Delete that workload.

**Cleanup:** run `./run.sh clean`. It deletes the `gpu-lab` namespace. Then run `terraform destroy`
in `../gcp/terraform`.

| Manifest | Pool it lands on | What to look at |
|---|---|---|
| `00-namespace.yaml` | — | Everything is in `gpu-lab`. |
| `01-gpu-smoke.yaml` | `l4` (g2-standard-4, selects `gpu-lab/pool: l4`) | The `probe.sh` output: `/dev/nvidia*`, the `/usr/local/nvidia` driver mount and the versions. Read it with `python -m gpurt.container --log out/gpu-smoke.log`. |
| `02-cuda-vectoradd.yaml` | `l4` | "Test PASSED": the CUDA runtime of the image and the driver of the node agree. |
| `03-nccl-tests-2gpu.yaml` | `l4x2` (g2-standard-24) | `nvidia-smi topo -m`, the transport of NCCL (`NCCL_DEBUG=INFO`) and busbw. Read them with `python -m gpurt.nccltests out/nccl-tests-2gpu.log`. The manifest pins nccl-tests to v2.20.0. |
| `04-time-sharing.yaml` | `l4-shared` | Two pods, one GPU UUID. Scale to 3. The third pod does not fit in the 2 advertised GPUs of the node. Thus the autoscaler adds a second node. The pool scales from 0 to `gpu_max_nodes` = 2. To see a pod *stay* Pending, scale past 2 × `gpu_max_nodes` or set `gpu_max_nodes = 1`. |
| `05-mig.yaml` | `a100-mig` | `nvidia-smi -L` lists a `MIG 1g.5gb Device` |
| `06-dcgm-alert-rules.yaml` | No pool (cluster-scoped) | GMP `ClusterRules` that `gpurt.dcgm.rules_manifest()` generates. Do not edit them by hand. |
| `alertmanager/alertmanager.example.yaml` | — | Where firing alerts go: routes for the managed Alertmanager of GMP. This file is not a Kubernetes manifest. Thus it is in its own folder, where `kubectl apply -f deploy/gke/` does not find it. |

```bash
$(terraform -chdir=../gcp/terraform output -raw get_credentials)
./run.sh status        # nodes, accelerator/sharing labels, allocatable nvidia.com/gpu
./run.sh smoke         # -> out/gpu-smoke.log
./run.sh vectoradd
./run.sh nccl          # -> out/nccl-tests-2gpu.log (first run: node scale-up + image pull, 10+ minutes)
./run.sh timeshare     # needs enable_time_sharing_pool = true
./run.sh mig           # needs enable_mig_pool = true and A100 quota
./run.sh rules         # DCGM alert rules (ClusterRules), then give Alertmanager receivers (below)
DRY_RUN=1 ./run.sh nccl   # print the kubectl commands only
```

When a Job fails, `run.sh` stops at once, because it polls for *Complete* or *Failed*. It still saves
the log of the Job.

Every GPU pod requests `nvidia.com/gpu` in `limits`. You cannot overcommit an extended resource. Thus
the requests are equal to the limits. Each GPU pod also selects an accelerator with
`cloud.google.com/gke-accelerator`, and it has a toleration for the `nvidia.com/gpu` taint. Primer §6
shows how the device plugin changes that request into device nodes and a driver mount. Layer 03 shows
how the scheduler places the pod
([`03-kubernetes-gpu/gpu-scheduling`](../../../../../03-kubernetes-gpu/gpu-scheduling)).

## DCGM metrics and alerts

With `enable_dcgm = true`, GKE runs a DCGM exporter, and Managed Prometheus scrapes it. In *Metrics
explorer*, change to PromQL. While `./run.sh nccl` runs, compare `DCGM_FI_PROF_SM_ACTIVE` with
`DCGM_FI_DEV_GPU_UTIL`. Notebook 06 explains what each field means. It also explains why `GPU_UTIL`
alone gives an incorrect picture.

**Which rules can fire depends on the exported fields** (`gpurt.dcgm.EXPORTED_BY`). The lab took
the data in `gpurt.dcgm.EXPORTED_BY` from the upstream files on 2026-09-26 (verify for your versions):

| Exporter | Missing fields | Rules that can never fire |
|---|---|---|
| stock dcgm-exporter (`default-counters.csv`) | `DCGM_FI_DEV_CLOCKS_EVENT_REASONS`. Also `DCGM_FI_PROF_SM_ACTIVE` and `SM_OCCUPANCY`, which are commented out. | `GpuThermalThrottling`, `GpuHardwareSlowdown`, `GpuBusyButUnderfilled` |
| GKE-managed DCGM (field list: verify). The GMP DCGM example from Google lists no XID, row-remap, clock-event or DRAM fields. | Probably the health fields | Probably every `GpuXid*`, row-remap and clock rule |
| self-managed exporter with [`../any-gpu/dcgm-counters.csv`](../any-gpu/dcgm-counters.csv) | none | none |

`python -c "from gpurt import dcgm; print(dcgm.rules_that_cannot_fire(open('fields.txt').read().split()))"`
shows the rules that cannot fire for any field list.

For the health rules on GKE, run a self-managed exporter with the counters CSV of the lab. An example
is the `prometheus-engine` `examples/nvidia-dcgm` DaemonSet from Google. It runs in the namespace
`gmp-public`, and a `PodMonitoring` scrapes it (verify). Replace its `counters.csv` ConfigMap with
`../any-gpu/dcgm-counters.csv`. Also disable the managed package, to prevent duplicate series (verify).

**Why `ClusterRules`:** on GMP, a namespaced `Rules` object evaluates only the metrics from its own
namespace. The exporter runs in a system namespace (or `gmp-public`), never in `gpu-lab`. The
cluster-scoped `ClusterRules` sees the whole cluster. The target labels of the scraper (`namespace`,
`pod`) have priority. Thus the workload labels of the exporter arrive as
`exported_namespace`/`exported_pod` (verify). The rules select on these labels.

**Where alerts go:** rules only evaluate. The rule-evaluator of GMP sends firing alerts to an
Alertmanager, not to Cloud Monitoring alerting policies. The managed Alertmanager reads its
configuration from a Secret in `gmp-public` (verify the name/key). Edit the receivers in
`alertmanager/alertmanager.example.yaml`. Then run this command:

```bash
kubectl -n gmp-public create secret generic alertmanager \
  --from-file=alertmanager.yaml=alertmanager/alertmanager.example.yaml --dry-run=client -o yaml | kubectl apply -f -
```

The alternative is Cloud Monitoring alerting policies with a PromQL condition (Terraform
`google_monitoring_alert_policy`, `conditions.condition_prometheus_query_language.query` in provider
8.4.0). These policies use the same expressions from `06-dcgm-alert-rules.yaml`.

Some items in the manifests have the mark `VERIFY:`. Examples are image tags, GKE node-label values
and MIG profiles. These items are product details. Before you rely on them, examine them again against the
current documentation.
