# deploy/gke — an MoE with expert parallelism on two L4s in GKE (T3, optional)

**What it does:** serves Qwen1.5-MoE-A2.7B-Chat (14.3 B parameters, 28.6 GB in bf16 — more than one
24 GB L4 holds) with vLLM on one `g2-standard-24` node (2 × L4, PCIe, no NVLink), as
`--tensor-parallel-size 2 --enable-expert-parallel` by default, then benchmarks it from a CPU Job
with `vllm bench serve`. `./run.sh layout` switches between TP, TP+EP and DP+EP on the same node,
which is notebook 04's comparison on managed hardware.

**No new Terraform.** The cluster is layer 02's:
[`02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/gcp/terraform/`](../../../../../02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/gcp/terraform/)
creates a zonal GKE Standard cluster whose `l4x2` pool (on by default, Spot, 0 → 2 nodes, GKE-managed
driver) is exactly the node these manifests select (`cloud.google.com/gke-accelerator: nvidia-l4`,
`node.kubernetes.io/instance-type: g2-standard-24`, a toleration for the `nvidia.com/gpu` taint).
Its [README](../../../../../02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/gcp/README.md) covers
quota, prerequisites and cost.

| File | What it is |
|---|---|
| `00-namespace.yaml` | namespace `moe-lab` (delete it and everything here is gone) |
| `01-vllm-moe-ep2.yaml` | `Deployment` (2 GPUs = TP × DP, probes on `/health`, 8 GiB `/dev/shm` for NCCL, HF cache) + `Service` on port 8000; `--enable-return-routed-experts` on, for notebook 02 |
| `02-bench-job.yaml` | `Job`: `vllm bench serve` against the Service (random 256-in / 128-out prompts, `--ignore-eos`); no GPU, so it runs on the system pool |
| `run.sh` | `status`, `up`, `layout tp\|tp_ep\|dp_ep`, `bench [CONCURRENCY]`, `logs`, `clean`; `DRY_RUN=1` prints the kubectl commands |

```bash
cd ../../../../../02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/gcp/terraform
terraform apply                                   # once; see its README (enable_multi_gpu_pool = true is the default)
$(terraform output -raw get_credentials)
cd -                                              # back here
./run.sh up                                       # 10-30 min the first time: Spot node from zero, image, 28.6 GB of weights
./run.sh bench 1 && ./run.sh bench 16             # -> out/bench_tp_ep_c1.txt, out/bench_tp_ep_c16.txt
./run.sh layout tp && ./run.sh bench 16           # the same model with every expert split in half
./run.sh layout dp_ep && ./run.sh bench 16        # data-parallel attention, experts split whole
python -m moelab bench-parse out/bench_tp_c16.txt
```

To read the router from inside the cluster, `kubectl -n moe-lab port-forward svc/vllm-moe 8000:8000`
and run notebook 02 with `MOELAB_URL=http://127.0.0.1:8000`: it finds Qwen1.5-MoE-A2.7B's 60 experts,
top-4, in `moelab.configs` by the served model id.

**What to look for.** `./run.sh logs` shows the memory each GPU loaded, the KV cache it got, and
*"Using default MoE config. Performance might be sub-optimal!"* — vLLM has no tuned fused-MoE config
for the L4 (notebook 05). Between layouts compare mean ITL at concurrency 1 (EP splits a token's
experts unevenly between the GPUs) and output tokens per second at 16 (notebook 04 predicts both).

**Driver.** The `vllm/vllm-openai:v0.30.0` image is a CUDA 13 build that needs an NVIDIA driver from
the 580 series (verify). Layer 02's pools install GKE's `DEFAULT` driver; if the pod fails with a
CUDA driver/runtime mismatch, set `gpu_driver_version = "LATEST"` in that lab's `terraform.tfvars`
and `terraform apply` again (new nodes get the newer driver).

## Cost and cleanup

The 2 × L4 node exists only while the vLLM pod does: a `g2-standard-24` on Spot costs a fraction of
its on-demand price (Spot is 60–91% off; see [`COMPUTE.md`](../../../../../COMPUTE.md) for dated
prices, verify), plus the cluster's system node and management fee. An hour of experiments is a few
dollars. The benchmark Job uses the system pool.

```bash
./run.sh clean                                    # delete the namespace; the l4x2 node scales to zero (~10 min)
terraform -chdir=../../../../../02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/gcp/terraform destroy
```

Spot capacity can be missing or reclaimed with ~30 s notice; the pod then restarts and re-downloads
the weights (the cache is node-local). For a steadier demo, create the pool without Spot
(`gpu_spot = false` in layer 02's tfvars).
