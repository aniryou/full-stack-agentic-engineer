# deploy/gke — GKE Inference Gateway, end to end (T3)

Runs on the cluster from [`../gcp/terraform`](../gcp/terraform/README.md). `install.sh` applies, in order:

| Step | Object(s) | File |
|---|---|---|
| 2 | InferencePool CRD (only if GKE does not already manage it) + InferenceObjective CRD | upstream release URLs |
| 3 | vLLM `Deployment` on an L4 Spot node (`Qwen/Qwen2.5-1.5B-Instruct`, verify) + `PodMonitoring` for its `/metrics` | `vllm.yaml`, `podmonitoring-vllm.yaml` |
| 4 | llm-d EPP (Helm `llm-d-router-gateway`, `provider.name=gke`) — also creates the `InferencePool` `vllm-qwen`, the `InferenceObjective`s, GKE's `HealthCheckPolicy`/`GCPBackendPolicy` and a PodMonitoring for EPP metrics | `epp-values.yaml` (what it renders: `rendered/`) |
| 5 | `Gateway` (`gke-l7-regional-external-managed`) + `HTTPRoute` → InferencePool | `gateway.yaml` |
| 6 | Custom Metrics Stackdriver Adapter (pinned commit, + a project IAM binding for its KSA) and the `HPA` on `vllm:num_requests_waiting` **and** `vllm:num_requests_running` | `hpa.yaml` |

```bash
DRY_RUN=1 PROJECT_ID=<id> ./install.sh     # read the plan first
PROJECT_ID=<id> ZONE=<zone> ./install.sh   # first vLLM start: node provisioning + image pull + weights, 5-15 min
kubectl get inferencepools,inferenceobjectives,gateway,httproute,hpa
```

Then send requests to the Gateway address (printed by the script), optionally with
`x-llm-d-inference-objective: premium|standard|batch`. `batch` has priority −10: while the pool is
saturated the EPP rejects it with 429 instead of queueing it.

Notes:

- `rendered/inferencepool.yaml` and `rendered/inferenceobjectives.yaml` are what the chart creates from
  `epp-values.yaml`; Helm applies them — they are kept for reading and for the offline schema check.
  With `router.inferencePool.create=false` the chart would switch the EPP to selector mode instead.
- The InferencePool's `failureMode` enum is `FailOpen | FailClose`; the chart README's "FailClosed"
  is rejected by the CRD.
- The HPA's metric names follow the adapter's convention for Managed Prometheus series
  (`prometheus.googleapis.com|<name>|<kind>`: the adapter README says to replace `/` with `|`; the
  `prometheus.googleapis.com/<name>/<kind>` metric type is marked VERIFY). `minReplicas` is 1:
  scale-to-zero needs the alpha `HPAScaleToZero` gate or KEDA with a router-side queue metric (notebook 03).
- The HPA has **two** metrics because queue-only collapses the pool at full load (notebook 03): the
  queue (`waiting`, target 5) reacts to bursts, occupied batch slots (`running`, target 24) hold
  capacity once the queue has drained. Both targets are derived from `--max-num-seqs=32` in
  `vllm.yaml` (24 = 75 % of the slots; 5 = Little's law with a 0.5 s queueing budget and an
  *assumed* ~3 s per request in the batch). Recalibrate both from a load test on your model and
  GPU — the derivation is in `hpa.yaml` and notebook 03 Exercise 3.4.
- `vllm.yaml` sets `--enable-prompt-tokens-details`, so responses carry
  `usage.prompt_tokens_details.cached_tokens` and `igwlab.bench` can report per-request hit rates.

**Cost:** see the Terraform README. With these workloads installed an *idle* hour still costs
~$0.44 (assumed prices, verify): `minReplicas: 1` keeps one vLLM pod, so one L4 Spot node stays up.
Only after `uninstall.sh` does the GPU pool drop to 0 (the system node and disks, ~$0.13–0.16/h, until
`terraform destroy`; `uninstall.sh` deletes the Gateway and with it the load balancer).
**Cleanup:** `PROJECT_ID=<id> ./uninstall.sh` (removes the workloads, the metrics adapter and its
project-level IAM binding, which `terraform destroy` would leave behind), then `terraform destroy`
in `../gcp/terraform`.
