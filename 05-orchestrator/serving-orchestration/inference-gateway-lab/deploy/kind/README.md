# deploy/kind — the real llm-d Router on a laptop (T0 + Docker, CPU only)

A one-node kind cluster with:

- the CRDs: `InferencePool` v1 (Gateway API Inference Extension v1.6.2) and `InferenceObjective`
  v1alpha2 (llm-d-router v0.10.0);
- three `llm-d-inference-sim` pods (`sim-deployment.yaml`: the fake backend's per-request latency
  formula and prefix cache, but no queueing of prefills behind each other — see `../local/README.md`);
- the **llm-d Router in standalone mode** from the `llm-d-router-standalone` Helm chart
  (`router-values.yaml`): the EPP plus an Envoy sidecar listening on :8081, an `InferencePool` named
  after the release (`igw`) selecting `app: vllm-sim`, and the objectives `premium` (100) and `batch` (−10).
  The EPP runs the lab preset `default-weighted` verbatim (`router.epp.pluginsCustomConfig`).

```bash
./up.sh                                               # DRY_RUN=1 ./up.sh to see every command first
kubectl port-forward svc/igw-epp 8081:8081 &
ROUTER_URL=http://localhost:8081 ../local/smoke.sh    # x-inference-pod shows which simulator served
kubectl logs deploy/igw-epp -c epp | tail             # the EPP's own view (JSON logs)
./down.sh                                             # deletes the kind cluster
```

Requirements: Docker, `kind` (v0.30+; the node image `kindest/node:v1.34.0` is marked verify),
`kubectl`, `helm` (v3.14+ for OCI charts). The chart's default EPP requests (8 CPU, 8 GiB) are reduced
in `router-values.yaml` so it fits on a laptop.

Versions are env-overridable: `GAIE_VERSION` (v1.6.2), `ROUTER_VERSION` (v0.10.0),
`ROUTER_CHART_VERSION` (defaults to `ROUTER_VERSION`; the llm-d guides use the floating channel `v0`).
All marked (verify): check the release pages before relying on them.

**Optional — Gateway mode on kind.** Instead of the standalone chart, install a Gateway API
implementation that supports InferencePool (the llm-d guides document agentgateway and Istio; verify
current versions), then the `llm-d-router-gateway` chart with `--set httpRoute.create=true` and an
HTTPRoute whose backendRef is the InferencePool. `deploy/gke/gateway.yaml` shows the object shapes
(swap the GatewayClass for your implementation's).

**Cost:** free (local CPU). **Cleanup:** `./down.sh`.
