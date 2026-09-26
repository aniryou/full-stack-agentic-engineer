# deploy/gke — GKE Inference Gateway, end to end (T3)

Runs on the cluster from [`../gcp/terraform`](../gcp/terraform/README.md). `install.sh` applies, in order:

| Step | Object(s) | File |
|---|---|---|
| 2 | InferencePool CRD (only if GKE does not already manage it) + InferenceObjective CRD | upstream release URLs |
| 3 | vLLM `Deployment` on an L4 Spot node (`Qwen/Qwen2.5-1.5B-Instruct`, verify) + `PodMonitoring` for its `/metrics` | `vllm.yaml`, `podmonitoring-vllm.yaml` |
| 4 | llm-d EPP (Helm `llm-d-router-gateway`, `provider.name=gke`) — also creates the `InferencePool` `vllm-qwen`, the `InferenceObjective`s, GKE's `HealthCheckPolicy`/`GCPBackendPolicy` and a PodMonitoring for EPP metrics | `epp-values.yaml` (what it renders: `rendered/`) |
| 5 | `Gateway` (`gke-l7-regional-external-managed`) + `HTTPRoute` → InferencePool | `gateway.yaml` |
| 6 | Custom Metrics Stackdriver Adapter (+ IAM for its KSA) and the `HPA` on `vllm:num_requests_waiting` | `hpa.yaml` |

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
- The HPA's metric name follows the adapter's convention for Managed Prometheus series
  (`prometheus.googleapis.com|<name>|<kind>`, marked VERIFY). `minReplicas` is 1: scale-to-zero needs
  the alpha `HPAScaleToZero` gate or KEDA with a router-side queue metric (see notebook 03).
- A second HPA metric on `vllm:num_requests_running` (target 6 of the batch slots) keeps capacity when
  the queue drains; notebook 03 shows why queue-only collapses. Add it once you have calibrated your
  engine's batch size.

**Cost:** see the Terraform README (~$0.44/h with one L4 Spot node, assumed prices, verify).
**Cleanup:** `./uninstall.sh`, then `terraform destroy` in `../gcp/terraform`.
