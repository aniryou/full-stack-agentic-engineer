# deploy/gcp — a quantized model on the serving lab's Cloud Run or GKE (T3, optional)

**Tier:** T3 (GCP, pay per use). This lab adds no infrastructure code: the Cloud Run service
(Terraform and a `gcloud` script) and the GKE manifests are the serving lab's,
[`../../../../serving-engine/vllm-serving-lab/deploy/gcp/`](../../../../serving-engine/vllm-serving-lab/deploy/gcp/).
What changes for a quantized model is the model id and a few `vllm serve` flags, so this directory
holds exactly that:

| File | What it is |
|---|---|
| `cloud-run.quantized.tfvars.example` | variables for the serving lab's `cloud-run/terraform`: a published INT4 AWQ model (default), FP8 quantized at load time, or your own checkpoint from `compress.sh` uploaded to Cloud Storage |
| `deploy-quantized.sh` | `SCHEME=bf16|fp8-online|w4a16` → runs the serving lab's `cloud-run/deploy.sh` with the right `MODEL`/`EXTRA_ARGS`, or (`TARGET=gke`) applies its `vllm.yaml` and patches the container args. `DRY_RUN=1` prints every command, including the `gcloud run deploy` line |

Both targets run **L4s** (sm_89): FP8 W8A8 runs natively, INT4 runs Marlin, and FP8 KV works through
FlashInfer — `python -m quantlab plan --gpu L4` is the table.

```bash
# Terraform (repeatable, IAM included)
cd ../../../../serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/terraform
cp ../../../../../quantization/quant-lab/deploy/gcp/cloud-run.quantized.tfvars.example terraform.tfvars   # edit project_id
terraform init && terraform apply

# or gcloud, through the wrapper
PROJECT_ID=my-project SCHEME=w4a16 KV_CACHE_DTYPE=fp8 ./deploy-quantized.sh
TARGET=gke SCHEME=fp8-online ./deploy-quantized.sh          # a cluster from the serving lab's gke/cluster.sh
```

Measure it the same way as locally: `gcloud run services proxy vllm-l4 --region us-central1 --port 8080 &`
then `QUANTLAB_URL=http://127.0.0.1:8080` for the notebooks, or `python -m quantlab bench --url http://127.0.0.1:8080`.

## What quantization changes here

Qwen2.5-1.5B-Instruct on the L4 at vLLM v0.30.0 defaults (`python -m quantlab kv --model qwen2.5-1.5b-instruct --gpu L4`;
the serving lab's memory model, an estimate until the startup log confirms it):

| | BF16 | INT4 (AWQ) | FP8 online |
|---|---|---|---|
| weights to load at cold start | 3.09 GB | 1.15 GB | 1.78 GB |
| KV blocks (16 tokens each) | 39,751 | 43,980 | 42,607 |
| concurrent 2,000-token sessions | 318 | 352 | 341 |

Smaller weights shorten the weight-download part of a Cloud Run cold start (the serving lab's
notebook 06 turns bytes into seconds) and leave more of the L4 for KV blocks; FP8 W8A8 also halves
prefill FLOP time on the L4. The accuracy gate is notebook 03's, on your own eval set.

## Cost and cleanup

Unchanged from the serving lab: Cloud Run bills the L4 instance per second while it exists and nothing
at `min_instances = 0`; GKE bills the cluster and the Spot L4 node while it runs (see the serving
lab's [`cloud-run/README.md`](../../../../serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/README.md)
and [`gke/README.md`](../../../../serving-engine/vllm-serving-lab/deploy/gcp/gke/README.md), prices in
[`COMPUTE.md`](../../../../../COMPUTE.md), verify).

```bash
terraform destroy                                   # Terraform path, in the serving lab's terraform dir
PROJECT_ID=my-project ./deploy-quantized.sh delete  # gcloud path
TARGET=gke ./deploy-quantized.sh delete             # GKE workload; the cluster: the serving lab's ./cluster.sh delete
gcloud storage rm -r gs://my-serving-lab-weights/Qwen2.5-1.5B-Instruct-FP8_DYNAMIC   # if you uploaded one
```
