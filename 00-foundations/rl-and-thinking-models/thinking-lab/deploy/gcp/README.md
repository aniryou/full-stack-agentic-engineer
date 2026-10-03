# deploy/gcp — the 04 serving lab's Google Cloud deploys, with a thinking model (T3)

**Tier:** T3 (GCP, pay per use, optional). This lab adds **no Terraform**. A thinking model on Google Cloud uses the
04 lab's deploy with different engine flags. The two files here hold those flags, one file for each target.
`tests/test_deploy.py` examines them offline:

| File | For | What changes versus the 04 lab's defaults |
|---|---|---|
| [`cloud-run-thinking.tfvars.example`](cloud-run-thinking.tfvars.example) | the 04 lab's Cloud Run Terraform, [`../../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/terraform/`](../../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/terraform/) | `Qwen/Qwen3-4B`, `max_model_len` 16384, `--reasoning-parser qwen3`, `--reasoning-config` for budgets, `--enable-prompt-tokens-details`, `--max-num-seqs 32`, `concurrency` 16, `request_timeout` 1800 s |
| [`gke/vllm-thinking.yaml`](gke/vllm-thinking.yaml), [`gke/podmonitoring-thinking.yaml`](gke/podmonitoring-thinking.yaml) | the 04 lab's GKE cluster ([`../../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/gke/`](../../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/gke/)) or layers 03/05's Terraform clusters | a `vllm-thinking` Deployment + Service on one L4 (Spot) with the same flags, and its Managed Prometheus scrape |

## Cloud Run (one L4, scale to zero)

```bash
LAB04=../../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/terraform
cp cloud-run-thinking.tfvars.example "$LAB04/thinking.tfvars"      # edit project_id, invoker_members
cd "$LAB04" && terraform init && terraform apply -var-file=thinking.tfvars
gcloud run services proxy vllm-thinking-l4 --region us-central1 --port 8080 &      # (verify the command in your gcloud)
THINKLAB_URL=http://127.0.0.1:8080 python -m thinklab ask "What is 47 * 23 - 318?"
```

First, read the 04 lab's [Cloud Run README](../../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/README.md).
It covers the paid billing account, the L4 quota, the private service and the cold-start arithmetic. Two things are
different for a thinking model:

* **`concurrency` is lower than for chat.** Each request holds thousands of KV tokens for a minute or more. Calculate
  the value with notebook 04: the KV pool of Qwen3-4B on an L4 (≈88K tokens, predicted), divided by the time-averaged
  context of your requests. Above that value, it is better that Cloud Run starts another instance than that it
  overfills this one.
* **`request_timeout` is longer.** A 4K-token trace at ~30-60 ms per token (simulated for an L4 under load) takes
  several minutes of streaming.

## GKE (L4 Spot)

```bash
# the cluster: the 04 lab's ./cluster.sh (gcloud), or the Terraform in layer 03 / 05
kubectl apply -f gke/vllm-thinking.yaml -f gke/podmonitoring-thinking.yaml
kubectl port-forward svc/vllm-thinking 8000:8000 &
THINKLAB_URL=http://127.0.0.1:8000 jupyter lab ../../notebooks        # notebooks 02-04 now measure the L4
```

In Cloud Monitoring (PromQL), these signals change with thinking:
`max(vllm:kv_cache_usage_perc)`, `sum(rate(vllm:num_preemptions_total[5m]))`,
`histogram_quantile(0.99, sum by (le) (rate(vllm:inter_token_latency_seconds_bucket[5m])))` and
`histogram_quantile(0.9, sum by (le) (rate(vllm:request_generation_tokens_bucket[5m])))`. Version v0.30.0 of vLLM
has no reasoning-specific metric. The per-request reasoning counts are in the
`usage.completion_tokens_details.reasoning_tokens` field of each response.

## Cost and cleanup

Cloud Run bills per second while an instance exists. With `min_instances = 0`, an idle service costs nothing. But
the next request then has a cold start (the image pull plus 8 GB of weights).
`terraform destroy -var-file=thinking.tfvars` removes the service. On GKE, a Spot `g2-standard-8` (one L4) costs a fraction of the
on-demand price (Spot is 60-91% off, verify in [`COMPUTE.md`](../../../../../COMPUTE.md)). You also pay for the
cluster:

```bash
kubectl delete -f gke/vllm-thinking.yaml -f gke/podmonitoring-thinking.yaml
# then the 04 lab's:  PROJECT_ID=my-project ./cluster.sh delete
```

The L4 pool scales back to zero about ten minutes after the Deployment is gone. The cluster continues to bill until
you delete it.
