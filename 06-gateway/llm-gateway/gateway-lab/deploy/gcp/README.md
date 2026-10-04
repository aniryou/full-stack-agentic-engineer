# deploy/gcp — the 04 and 05 labs' Google Cloud deploys as the gateway's upstreams (T3)

**Tier:** T3 (GCP, pay per use, optional). This lab adds **no Terraform**. A gateway on Google Cloud is the same
process in front of upstreams that layers 04 and 05 already deploy. The configuration changes: the `base_url` of the
provider, its credential and the upstream model name. The location where the gateway itself runs also changes.

| Upstream | Deployed by | Provider entry in the gateway config |
|---|---|---|
| vLLM on Cloud Run, one L4, scale to zero | [`04-…/vllm-serving-lab/deploy/gcp/cloud-run/`](../../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/README.md) (Terraform or `deploy.sh`) | `dialect: openai`, `base_url:` the service URL (or a local `gcloud run services proxy`), `upstream:` the `served_model_name` (default: the model id) |
| vLLM on GKE (L4 Spot) | [`04-…/vllm-serving-lab/deploy/gcp/gke/`](../../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/gke/README.md) | `base_url:` the Service (`vllm`, port 8000) seen from inside the cluster, or a `kubectl port-forward` |
| GKE Inference Gateway in front of a vLLM pool | [`05-…/inference-gateway-lab/deploy/gcp/terraform/`](../../../../../05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/gcp/terraform/README.md) + [`deploy/gke/`](../../../../../05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/gke/README.md) | `base_url:` the address of the `inference-gateway` Gateway (port 80), `upstream: qwen`. The endpoint picker selects the pod. |

The last row shows the split that this topic is about. **This gateway selects the pool**, and it falls back to
another provider when the pool is down or saturated. **The 05 endpoint picker selects the replica** inside the pool
(05 PRIMER §1.3, §7). A 429 from the Inference Gateway (a sheddable `InferenceObjective`, rejected at saturation)
is a fall-through error here, so the chain moves to its next target.

## Pointing the gateway at a T3 upstream

Here is a config for the Cloud Run row, next to `gwlab/configs/vllm.yaml`:

```yaml
providers:
  cloudrun:
    dialect: openai
    base_url: ${CLOUD_RUN_URL:-http://127.0.0.1:8081}   # `gcloud run services proxy vllm-l4 --port 8081` (verify)
    region: us
    first_byte_timeout_s: 120                             # a cold start after scale-to-zero takes minutes
    cache_salt: true
    otel_name: vllm
models:
  cloudrun/qwen: {provider: cloudrun, upstream: Qwen/Qwen2.5-1.5B-Instruct, context_window: 8192, tools: false,
                  price: {gpu_hour_usd: 0.70, tokens_per_s: 300, utilisation: 0.3}}    # (verify) L4 price; your measurement
aliases:
  chat: {policy: ordered, targets: [cloudrun/qwen, acme/fast]}
```

The service of the 04 lab is private. From a laptop, `gcloud run services proxy` adds your identity to each request
(the gateway then talks to `127.0.0.1`). The alternative for a gateway that runs on Google Cloud is a Google-signed
ID token whose audience is the service URL (verify). The provider credential of this lab is a static key. Thus that
path is a follow-up, and the lab does not implement it.

Keep `first_byte_timeout_s` above the cold start. The cold start is the image pull plus the weights plus the engine
start, and the 04 lab's
[notebook 06](../../../../../04-inference-engine/serving-engine/vllm-serving-lab/notebooks/06_deploy_on_cloud_run_gpu.ipynb)
calculates it. As an alternative, set `min_instances = 1` in the 04 lab's Cloud Run deploy.

## Running the gateway itself on Cloud Run (CPU)

The gateway is a CPU service. This section gives the shape, not a tested script (all verify). The gateway needs
three things that the local stack does not need:

- A config that is in the image.
- A listener on `0.0.0.0`. By default, the CLI binds `127.0.0.1`, and Cloud Run cannot reach that address.
- An image. The lab root has no Dockerfile. Thus, if you use `--source .`, the deploy falls back to buildpacks with
  no entrypoint.
  Build [`deploy/local/Dockerfile`](../local/Dockerfile) instead.

```bash
# 1. a config baked into the image: copy gwlab/configs/vllm.yaml to gwlab/configs/cloudrun.yaml, replace its
#    `local` provider and `local/llm` model with the `cloudrun` entries above, point `chat` at cloudrun/qwen, and
#    point (or drop) the `acme` fallback -- there is no fake provider on Cloud Run
# 2. build and push the lab image (build context = the lab root; the Dockerfile's WORKDIR is /lab)
IMAGE=us-central1-docker.pkg.dev/${PROJECT}/gwlab/gateway:0.1
docker build -f deploy/local/Dockerfile -t "${IMAGE}" . && docker push "${IMAGE}"
# 3. run the gateway process (not the image's default `--help`), on all interfaces, on the port Cloud Run routes to
gcloud run deploy gwlab-gateway --image "${IMAGE}" --region us-central1 --no-allow-unauthenticated --port 8080 \
  --command python --args=-m,gwlab,gateway,--host,0.0.0.0,--port,8080 \
  --set-env-vars GWLAB_CONFIG=/lab/gwlab/configs/cloudrun.yaml,CLOUD_RUN_URL=https://<the 04 lab's service URL> \
  --set-secrets GWLAB_ADMIN_TOKEN=gwlab-admin:latest,GWLAB_SALT_SECRET=gwlab-salt:latest \
  --timeout 3600 --concurrency 250 --min-instances 0
```

Know three things before you do this:

- The ledger and keys are **sqlite in one process**. On Cloud Run, they go away when the instance stops, and
  the instances do not share them. Thus a real deployment moves them to a database, and the buckets to Redis (scaling
  primer §5.1, §5.9).
- A streamed request counts against the request timeout (default 5 minutes, maximum 60, scaling primer §5.7).
- Spans go to Cloud Trace over OTLP (`telemetry.googleapis.com`, scaling primer §9) when
  `OTEL_EXPORTER_OTLP_ENDPOINT` has a value and the `otlp` extra is installed.

## Cost and cleanup

The gateway on Cloud Run with `--min-instances 0` costs nothing when idle, and cents per hour of traffic (verify).
Most of the cost comes from the GPU upstreams. An L4 on Cloud Run has per-second billing while an instance exists.
The GKE stack of the 05 lab keeps one L4 Spot node up while it is installed (its README gives about $0.44/h,
verify).

Clean up each upstream in its own way:

- In the 04 lab's `cloud-run/terraform`, run `terraform destroy`.
- In the 05 lab, run `./uninstall.sh`, then `terraform destroy`.

For the gateway, run `gcloud run services delete gwlab-gateway --region us-central1`. For prices and quotas, see
[`COMPUTE.md`](../../../../../COMPUTE.md).
