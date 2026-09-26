# Build status — layers 01–05 (+ deep primers)

Legend: `building` (agents writing) → `built` (builder validation passed) → `reviewed` (adversarial review + fixes passed) → `merged` (on `main`).
WIP snapshots are pushed to `claude/gifted-johnson-9gjwzc` (draft PR). Reviewed layers are merged to `main` through their own branch/PR.

| Layer / item | Paths | State | Next step |
|---|---|---|---|
| 01 roofline-and-fabric | `01-hardware-gpu-fabric/roofline-and-fabric/` (PRIMER, `roofline-core`, `gpu-bench-lab`) | PRIMER+core built (54 tests, 4/4 nbs); lab building | lab report → review workflow for the whole layer |
| 02 cuda-and-nccl | `02-cuda-nccl-runtime/cuda-and-nccl/` (PRIMER, `cuda-nccl-core`, `cuda-nccl-lab`) | building | builder reports → review workflow |
| 03 gpu-scheduling | `03-kubernetes-gpu/gpu-scheduling/` (PRIMER, `k8s-gpu-core`, `k8s-gpu-lab`) | PRIMER+core built (33 tests, 5/5 nbs); lab building | lab report → review workflow for the whole layer |
| 04 serving-engine | `04-inference-engine/serving-engine/` (PRIMER, `mini-engine-core`, `vllm-serving-lab`) | building | builder reports → review workflow |
| 05 serving-orchestration | `05-orchestrator/serving-orchestration/` (PRIMER, `orchestrator-core`, `inference-gateway-lab`) | building | builder reports → review workflow |
| vLLM internals primer | `04-inference-engine/vllm-internals/` | building | review → merge with layer 04 |
| FlashAttention deep dive | `04-inference-engine/flash-attention/flash-attention-deep-dive.md` | building | review → merge with layer 04 |
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
