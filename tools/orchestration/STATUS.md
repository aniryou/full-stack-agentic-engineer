# Build status — layers 01–05 (+ deep primers)

Legend: `building` (agents writing) → `built` (builder validation passed) → `reviewed` (adversarial review + fixes passed) → `merged` (on `main`).
WIP snapshots are pushed to `claude/gifted-johnson-9gjwzc` (draft PR). Reviewed layers are merged to `main` through their own branch/PR.

| Layer / item | Paths | State | Next step |
|---|---|---|---|
| 01 roofline-and-fabric | `01-hardware-gpu-fabric/roofline-and-fabric/` (PRIMER, `roofline-core`, `gpu-bench-lab`) | PRIMER+core REVIEWED ✓ (25 findings incl. 2 blocking fixed; 58 tests); lab built (58 tests, 4/4 nbs, TF valid), review running | both reviews pass → integrate layer README → merge layer 01 to main |
| 02 cuda-and-nccl | `02-cuda-nccl-runtime/cuda-and-nccl/` (PRIMER, `cuda-nccl-core`, `cuda-nccl-lab`) | PRIMER+core REVIEWED ✓ (23 findings fixed, 128 tests); lab built (92 tests, 6/6 nbs, TF valid), review running | both reviews pass → integrate layer README → merge layer 02 to main |
| 03 gpu-scheduling | `03-kubernetes-gpu/gpu-scheduling/` (PRIMER, `k8s-gpu-core`, `k8s-gpu-lab`) | PRIMER+core REVIEWED ✓ (27 findings incl. 3 blocking fixed + 1 by validator; 49 tests); lab built (97 tests, 4/4 nbs, TF valid, 23 CRD objects checked), review running | both reviews pass → integrate layer README → merge layer 03 to main |
| 04 serving-engine | `04-inference-engine/serving-engine/` (PRIMER, `mini-engine-core`, `vllm-serving-lab`) | PRIMER+core built (59 tests, 6/6 nbs), review running; lab built (58 tests, 6/6 nbs, TF valid), review running | both reviews + vLLM/FA primers pass → integrate layer README → merge layer 04 to main |
| 05 serving-orchestration | `05-orchestrator/serving-orchestration/` (PRIMER, `orchestrator-core`, `inference-gateway-lab`) | PRIMER+core built (43 tests, 5/5 nbs), review running; lab built (64 tests, 5/5 nbs, TF valid), review running | both reviews pass → integrate layer README → merge layer 05 to main |
| vLLM internals primer | `04-inference-engine/vllm-internals/` (primer 1,416 lines, source-map, 1 notebook) | built; review running | merge with layer 04 |
| FlashAttention deep dive | `04-inference-engine/flash-attention/` (deep-dive.md, fa_calculators.py + 49 tests, deep_dive notebook; practice notebook repaired) | REVIEWED ✓ (27 findings incl. 4 blocking fixed) | Colab-inject the new notebook at integration; merge with layer 04 |
| Root docs | `CURRICULUM.md`, `COMPUTE.md` | built | reconcile with what was actually built, then merge last |
| Integration | layer READMEs, root README, `CLAUDE.md` decisions log, Colab links (`tools/gen_colab_index.py`) | pending | after each layer's review; final pass at the end |

## Resuming after an interruption
1. `git checkout claude/gifted-johnson-9gjwzc && git pull` — the latest WIP snapshot.
2. Read this file, then `SPEC.md` §7 for the report format builders/reviewers use.
3. For any layer still `building`: run its validation (SPEC §4) to see what state the tree is in; finish or re-launch a builder with the SPEC §6 block for that layer.
4. For `built` layers: run the review workflow (two adversarial reviewers — concepts and runnability — then a fixer), then integrate and merge.

## Notes for the review / integration passes
- Layer 04 lab, free-T4 path: vLLM on Turing (compute capability 7.5) needs `--dtype half` and a non-FlashAttention backend; confirm the pinned vLLM
  release still supports 7.5 and say so in the README `(verify)`. Kaggle's P100 (capability 6.0) is below vLLM's minimum — the Kaggle recipe must pick "GPU T4 x2".
- Root `README.md` and the layer READMEs still describe 02/03/05 as empty; the integration pass rewrites them and regenerates Colab links.
- `.gitignore` now excludes `terraform.tfvars` / `*.auto.tfvars` (COMPUTE.md tells learners to check with `git check-ignore`).
- vLLM `main` (commit 5840d95, 2026-09-25; PyPI 0.30.0): Model Runner V2 and async scheduling are default-on; `VLLM_USE_V1` and `VLLM_ATTENTION_BACKEND` were removed. The layer-04 lab review must check the lab's env vars/flags against this.
- After the vLLM-internals review completes, check two items the FA validator flagged in `vllm-internals-primer.md`: §6.4 uses a 57× figure (verify against the FA deep dive §8.1: 71.1×/56.9×), and §6.3 omits the FA3-on-SM90 / FA4 condition for FP8 KV cache support.
- After the 03 lab review completes, check k8s-gpu-lab (kindsim predictor, notebooks) does not assert that time-sliced replicas share one GPU on a fresh node: the real NVIDIA device plugin's distributed allocation takes replicas from the least-loaded GPUs (fixed in the core: `gpusched/deviceplugin.py`). Core API also changed: `startup_latency(pull_GBps=, load_GBps=)`, `simulate(..., delay_after_add_s, spot_rate_per_node_hr)` — the lab does not import the core, but check any copied snippets.
