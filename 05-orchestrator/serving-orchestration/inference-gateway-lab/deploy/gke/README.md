# deploy/gke — GKE Inference Gateway, end to end (T3)

**What it does.** It puts GKE Inference Gateway in front of vLLM, on the cluster from
[`../gcp/terraform`](../gcp/terraform/README.md). It installs these objects: an InferencePool with the llm-d endpoint
picker, InferenceObjective priorities, a regional Gateway, and an HPA on Managed Prometheus metrics.

**Cost and cleanup, in short.** While the workloads are installed, the cost is ~$0.44/h, also with no traffic,
because one L4 Spot node stays up (assumed prices, verify). For the cleanup, run `PROJECT_ID=<id> ./uninstall.sh`,
then run `terraform destroy` in `../gcp/terraform`. The details are at the end.

`install.sh` applies these objects, in this order:

| Step | Object(s) | File |
|---|---|---|
| 2 | The InferencePool CRD (only if GKE does not already manage it), and the InferenceObjective CRD | upstream release URLs |
| 3 | A vLLM `Deployment` on an L4 Spot node (`Qwen/Qwen2.5-1.5B-Instruct`, verify), and a `PodMonitoring` for its `/metrics` | `vllm.yaml`, `podmonitoring-vllm.yaml` |
| 4 | The llm-d EPP (Helm `llm-d-router-gateway`, `provider.name=gke`). It also makes the `InferencePool` `vllm-qwen`, the `InferenceObjective`s, the `HealthCheckPolicy`/`GCPBackendPolicy` of GKE and a PodMonitoring for the EPP metrics | `epp-values.yaml` (what it renders: `rendered/`) |
| 5 | A `Gateway` (`gke-l7-regional-external-managed`), and an `HTTPRoute` to the InferencePool | `gateway.yaml` |
| 6 | The Custom Metrics Stackdriver Adapter (a pinned commit, with a project IAM binding for its KSA). The `HPA` on `vllm:num_requests_waiting` **and** `vllm:num_requests_running` | `hpa.yaml` |

```bash
DRY_RUN=1 PROJECT_ID=<id> ./install.sh     # read the plan first
PROJECT_ID=<id> ZONE=<zone> ./install.sh   # first vLLM start: node provisioning + image pull + weights, 5-15 min
kubectl get inferencepools,inferenceobjectives,gateway,httproute,hpa
```

Then send requests to the Gateway address. The script prints this address. As an option, add the header
`x-llm-d-inference-objective: premium|standard|batch`. `batch` has priority −10. While the pool is saturated, the
EPP rejects a `batch` request with 429, and does not put it in a queue.

Notes:

- `rendered/inferencepool.yaml` and `rendered/inferenceobjectives.yaml` are what the chart makes from
  `epp-values.yaml`. Helm applies them. The repository keeps them for you to read and for the offline schema check.
  With `router.inferencePool.create=false`, the chart puts the EPP in selector mode instead.
- The `failureMode` enum of the InferencePool is `FailOpen | FailClose`. The chart README says "FailClosed", and the
  CRD rejects that value.
- The metric names of the HPA obey the convention of the adapter for Managed Prometheus series
  (`prometheus.googleapis.com|<name>|<kind>`). The adapter README says to replace `/` with `|`. The
  `prometheus.googleapis.com/<name>/<kind>` metric type has the mark VERIFY. `minReplicas` is 1,
  because scale-to-zero needs the alpha `HPAScaleToZero` gate, or KEDA with a queue metric on the router side (notebook 03).
- The HPA has **two** metrics, because an HPA with only the queue metric collapses the pool at full load (notebook 03). The
  queue (`waiting`, target 5) reacts to bursts. The occupied batch slots (`running`, target 24) hold the capacity
  after the queue is empty. The two targets come from `--max-num-seqs=32` in `vllm.yaml`. 24 is 75 % of the slots.
  5 comes from Little's law with a queue budget of 0.5 s and an *assumed* ~3 s for each request in the batch.

  Calibrate the two targets again from a load test on your model and GPU. The derivation is in `hpa.yaml` and in
  notebook 03 Exercise 3.4.
- `vllm.yaml` sets `--enable-prompt-tokens-details`. Thus the responses carry
  `usage.prompt_tokens_details.cached_tokens`, and `igwlab.bench` can show the hit rate of each request.

**Cost:** See the Terraform README. With these workloads installed, an hour with *no traffic* still costs ~$0.44
(assumed prices, verify). One L4 Spot node stays up, because `minReplicas: 1` keeps one vLLM pod. The GPU pool goes
down to 0 only after `uninstall.sh`. Until `terraform destroy`, you pay for the system node and the disks
(~$0.13–0.16/h). `uninstall.sh` deletes the Gateway, and with it the load balancer.

**Cleanup:** Run `PROJECT_ID=<id> ./uninstall.sh`. It removes the workloads, the metrics adapter and the
project-level IAM binding of the adapter. `terraform destroy` does not remove that binding. Then run `terraform destroy`
in `../gcp/terraform`.
