# deploy/gcp/gke — vLLM on an L4 node pool, scraped by Managed Prometheus (T3)

**Tier:** T3 (GCP). One vLLM replica on a Spot L4 node that the autoscaler creates on demand, and a
`PodMonitoring` so the same `vllm:*` series this lab parses locally land in Cloud Monitoring.
Routing across replicas and autoscaling on engine signals are the next layer:
[`05-orchestrator/serving-orchestration/`](../../../../../../05-orchestrator/serving-orchestration/).

This directory is the minimal **gcloud** path: one script and two manifests keep the engine the
subject. The same kind of cluster as **Terraform** — zonal GKE Standard, an L4 Spot pool that scales
from zero with a GKE-installed driver, managed Prometheus — lives in layer 03's lab,
[`k8s-gpu-lab/deploy/gcp/terraform/`](../../../../../../03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/deploy/gcp/terraform/)
(plus DWS flex-start and GCS FUSE), and in layer 05's lab,
[`inference-gateway-lab/deploy/gcp/terraform/`](../../../../../../05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/gcp/terraform/)
(plus the Gateway API and a proxy-only subnet); this lab's own Terraform is the Cloud Run service in
[`../cloud-run/terraform/`](../cloud-run/terraform/). The manifests here apply to either cluster.

| File | What it is |
|---|---|
| `cluster.sh` | zonal GKE Standard cluster + L4 **Spot** pool autoscaling **0..2** (driver installed by GKE), then applies the manifests. `DRY_RUN=1` prints the commands |
| `vllm.yaml` | `Deployment` (1 GPU, probes on `/health`, `/dev/shm`, HF cache) + `Service` on port 8000 |
| `podmonitoring.yaml` | `monitoring.googleapis.com/v1` `PodMonitoring`: scrape `/metrics` every 15 s |

```bash
PROJECT_ID=my-project ./cluster.sh           # ~10 min: cluster, node pool, then the pod waits for a Spot L4
kubectl port-forward svc/vllm 8000:8000 &
python -m servelab bench --url http://127.0.0.1:8000 --rate 4 -n 100 --slo-ttft-ms 1000 --slo-tpot-ms 60
```

## Reading the engine in Cloud Monitoring

Managed Prometheus stores the scraped series under their Prometheus names; query them with PromQL
in Metrics Explorer or Grafana. The queries (`servelab.metrics.PROMQL`):

| Question | PromQL |
|---|---|
| prefix-cache hit rate | `sum(rate(vllm:prefix_cache_hits_total[5m])) / sum(rate(vllm:prefix_cache_queries_total[5m]))` |
| TTFT p99 (interpolated in buckets) | `histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))` |
| queue depth | `sum(vllm:num_requests_waiting)` |
| KV pressure | `max(vllm:kv_cache_usage_perc)` |
| preemptions per second | `sum(rate(vllm:num_preemptions_total[5m]))` |

Two cautions from notebook 02 apply unchanged: a percentile from these histograms is an
interpolation inside vLLM's bucket edges (the first queue-time bucket is 0-0.3 s, so a "p50 queue
time" of 150 ms can mean "no queue at all"), and `rate()` over a window is the only honest way to
read counters. `vllm:num_requests_waiting` and `vllm:kv_cache_usage_perc` are the signals an
autoscaler should use (layer 05) — GPU utilization is not.

## Cost and cleanup

A `g2-standard-8` (1 × L4, 8 vCPU, 32 GB) on Spot costs a fraction of the ~$0.7-1/hr on-demand
L4 VM price (Spot is 60-91% off; verify current prices in [`COMPUTE.md`](../../../../../../COMPUTE.md)),
plus the cluster: one `e2-standard-4` system node and the GKE cluster fee (the free tier covers
one zonal cluster per billing account; verify). The L4 pool scales back to zero ~10 minutes after
the Deployment is gone; the cluster keeps billing until you delete it:

```bash
kubectl delete -f vllm.yaml -f podmonitoring.yaml
PROJECT_ID=my-project ./cluster.sh delete
```

Spot capacity can be unavailable or reclaimed with ~30 s notice; for a steady demo drop the
`cloud.google.com/gke-spot` selector and create the pool without `--spot`.
