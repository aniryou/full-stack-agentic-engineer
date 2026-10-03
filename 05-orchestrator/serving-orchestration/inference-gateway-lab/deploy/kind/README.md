# deploy/kind — the real llm-d Router on a laptop (T0 + Docker, CPU only)

A kind cluster with one node, which has these parts:

- The CRDs: `InferencePool` v1 (Gateway API Inference Extension v1.6.2) and `InferenceObjective`
  v1alpha2 (llm-d-router v0.10.0).
- Three `llm-d-inference-sim` pods (`sim-deployment.yaml`). They have the per-request latency formula and the
  prefix cache of the fake backend. But they do not put prefills in a queue behind each other (see
  `../local/README.md`).
- The **llm-d Router in standalone mode**, from the `llm-d-router-standalone` Helm chart (`router-values.yaml`).
  It contains the EPP with an Envoy sidecar that listens on :8081, and an `InferencePool`. The pool has the name of
  the release (`igw`) and selects `app: vllm-sim`. It also contains the objectives `premium` (100) and `batch` (−10).
  The EPP runs the lab preset `default-weighted` verbatim (`router.epp.pluginsCustomConfig`).

**Cost:** free (local CPU). **Cleanup:** run `./down.sh`.

```bash
./up.sh                                               # DRY_RUN=1 ./up.sh to see every command first
kubectl port-forward svc/igw-epp 8081:8081 &
ROUTER_URL=http://localhost:8081 ../local/smoke.sh    # x-inference-pod shows which simulator served
kubectl logs deploy/igw-epp -c epp | tail             # the EPP's own view (JSON logs)
./down.sh                                             # deletes the kind cluster
```

Requirements: Docker, `kind` (v0.30+, the node image `kindest/node:v1.34.0` has the mark verify),
`kubectl` and `helm` (v3.14+ for OCI charts). `router-values.yaml` decreases the default EPP requests of the
chart (8 CPU, 8 GiB), so that the EPP fits on a laptop.

You can override the versions with environment variables: `GAIE_VERSION` (v1.6.2), `ROUTER_VERSION` (v0.10.0) and
`ROUTER_CHART_VERSION`. `ROUTER_CHART_VERSION` has the default `ROUTER_VERSION`. The llm-d guides use the floating
channel `v0`. All of these have the mark (verify). Examine the release pages before you trust them.

**Optional: Gateway mode on kind.** Instead of the standalone chart, install a Gateway API
implementation that supports InferencePool. The llm-d guides document agentgateway and Istio (verify the current versions).
Then install the `llm-d-router-gateway` chart with `--set httpRoute.create=true`, and an HTTPRoute whose backendRef
is the InferencePool. `deploy/gke/gateway.yaml` shows the shapes of the objects. Replace the GatewayClass with the
GatewayClass of your implementation.
