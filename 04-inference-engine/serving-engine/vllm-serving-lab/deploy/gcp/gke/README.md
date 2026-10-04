# deploy/gcp/gke — vLLM on an L4 node pool, scraped by Managed Prometheus (T3)

**Tier:** T3 (GCP). You get one vLLM replica on a Spot L4 node. The autoscaler creates that node on
demand. You also get a `PodMonitoring`. It makes sure that the same `vllm:*` series that this lab
parses locally go into Cloud Monitoring. Routing across replicas and autoscaling on engine signals
belong to the next layer:
[`05-orchestrator/serving-orchestration/`](../../../../../../05-orchestrator/serving-orchestration/).

This directory is the minimal **gcloud** path. It has one script and two manifests, so that the
engine stays the subject.

The same kind of cluster is also available as **Terraform**. It is a zonal GKE Standard cluster with
managed Prometheus. Its L4 Spot pool scales from zero and has a GKE-installed driver. Layer 03's
lab has Terraform for this kind of cluster in
[`k8s-gpu-lab/deploy/gcp/terraform/`](../../../../../../03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/deploy/gcp/terraform/) (plus DWS flex-start and GCS FUSE).
Layer 05's lab also has Terraform for it in [`inference-gateway-lab/deploy/gcp/terraform/`](../../../../../../05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/gcp/terraform/) (plus
the Gateway API and a proxy-only subnet).

The Terraform of this lab is the Cloud Run service in
[`../cloud-run/terraform/`](../cloud-run/terraform/). The manifests here apply to the cluster of either lab.

| File | What it is |
|---|---|
| `cluster.sh` | A zonal GKE Standard cluster and an L4 **Spot** pool that autoscales **0..2** (GKE installs the driver). Then the script applies the manifests. `DRY_RUN=1` prints the commands. |
| `vllm.yaml` | A `Deployment` (1 GPU, probes on `/health`, `/dev/shm`, HF cache) and a `Service` on port 8000. |
| `podmonitoring.yaml` | A `monitoring.googleapis.com/v1` `PodMonitoring`. It scrapes `/metrics` every 15 s. |

```bash
PROJECT_ID=my-project ./cluster.sh           # ~10 min: cluster, node pool, then the pod waits for a Spot L4
kubectl port-forward svc/vllm 8000:8000 &
python -m servelab bench --url http://127.0.0.1:8000 --rate 4 -n 100 --slo-ttft-ms 1000 --slo-tpot-ms 60
```

## Reading the engine in Cloud Monitoring

Managed Prometheus stores the scraped series under their Prometheus names. Query them with PromQL
in Metrics Explorer or Grafana. The queries (`servelab.metrics.PROMQL`) are:

| Question | PromQL |
|---|---|
| prefix-cache hit rate | `sum(rate(vllm:prefix_cache_hits_total[5m])) / sum(rate(vllm:prefix_cache_queries_total[5m]))` |
| TTFT p99 (interpolated in buckets) | `histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))` |
| queue depth | `sum(vllm:num_requests_waiting)` |
| KV pressure | `max(vllm:kv_cache_usage_perc)` |
| preemptions per second | `sum(rate(vllm:num_preemptions_total[5m]))` |

Two cautions from notebook 02 apply without change. First, a percentile from these histograms is
an interpolation inside vLLM's bucket edges. The first queue-time bucket is 0-0.3 s. Thus a "p50
queue time" of 150 ms can mean "no queue at all". Second, `rate()` over a window is the only correct
way to read counters.

The correct signals for an autoscaler are `vllm:num_requests_waiting` and
`vllm:kv_cache_usage_perc` (layer 05). GPU utilization is not a correct signal.

## Cost and cleanup

A `g2-standard-8` (1 × L4, 8 vCPU, 32 GB) on Spot costs a fraction of the ~$0.7-1/hr on-demand
L4 VM price. Spot is 60-91% off (verify: the current prices are in
[`COMPUTE.md`](../../../../../../COMPUTE.md)). You also pay for the cluster: one
`e2-standard-4` system node and the GKE cluster fee. The free tier covers one zonal cluster per
billing account (verify).

The L4 pool scales back to zero ~10 minutes after the Deployment is gone. But you pay for the
cluster until you delete it:

```bash
kubectl delete -f vllm.yaml -f podmonitoring.yaml
PROJECT_ID=my-project ./cluster.sh delete
```

Spot capacity can be unavailable. GCP can also take it back with ~30 s notice. For a steady demo,
remove the `cloud.google.com/gke-spot` selector. Then create the pool without `--spot`.
