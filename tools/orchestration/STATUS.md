# Build status — layers 01–05 (+ deep primers)

Legend: `building` (agents writing) → `built` (builder validation passed) → `reviewed` (adversarial review + fixes passed) → `merged` (on `main`).
WIP snapshots are pushed to `claude/gifted-johnson-9gjwzc` (draft PR). Reviewed layers are merged to `main` through their own branch/PR.

| Layer / item | Paths | State | Next step |
|---|---|---|---|
| 01 roofline-and-fabric | `01-hardware-gpu-fabric/roofline-and-fabric/` (PRIMER, `roofline-core`, `gpu-bench-lab`) | PRIMER+core REVIEWED ✓ (25 findings, 58 tests); lab REVIEWED ✓ (40 findings, 89 tests) | integrating on branch claude/gifted-johnson-9gjwzc-l01 → PR → merge to main |
| 02 cuda-and-nccl | `02-cuda-nccl-runtime/cuda-and-nccl/` (PRIMER, `cuda-nccl-core`, `cuda-nccl-lab`) | PRIMER+core REVIEWED ✓ (23 findings fixed, 128 tests); lab built (92 tests, 6/6 nbs, TF valid), review running | both reviews pass → integrate layer README → merge layer 02 to main |
| 03 gpu-scheduling | `03-kubernetes-gpu/gpu-scheduling/` (PRIMER, `k8s-gpu-core`, `k8s-gpu-lab`) | PRIMER+core REVIEWED ✓ (27 findings, 49 tests); lab REVIEWED ✓ (34 findings, 118 tests) | integrating on branch claude/gifted-johnson-9gjwzc-l03 (stacked on -l01) → PR → merge |
| 04 serving-engine | `04-inference-engine/serving-engine/` (PRIMER, `mini-engine-core`, `vllm-serving-lab`) | PRIMER+core REVIEWED ✓ (29 findings incl. 1 blocking fixed; 67 tests); lab built (58 tests, 6/6 nbs, TF valid), review running | lab review passes → integrate (04 README incl. vllm-internals + FA deep dive) → merge layer 04 |
| 05 serving-orchestration | `05-orchestrator/serving-orchestration/` (PRIMER, `orchestrator-core`, `inference-gateway-lab`) | PRIMER+core built (43 tests, 5/5 nbs), review running; lab built (64 tests, 5/5 nbs, TF valid), review running | both reviews pass → integrate layer README → merge layer 05 to main |
| vLLM internals primer | `04-inference-engine/vllm-internals/` (primer, source-map, 1 notebook) | REVIEWED ✓ (34 findings incl. 9 blocking fixed; ~60 source refs verified at main@5840d95) | integrator: list in 04 README + Colab index; merge with layer 04 |
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
- DONE by the vLLM review validator: §6.4 71.1× like-for-like figure added; §6.3 FP8-KV condition (FA3 on SM90 / FA4) added.
- After the 03 lab review completes, check k8s-gpu-lab (kindsim predictor, notebooks) does not assert that time-sliced replicas share one GPU on a fresh node: the real NVIDIA device plugin's distributed allocation takes replicas from the least-loaded GPUs (fixed in the core: `gpusched/deviceplugin.py`). Core API also changed: `startup_latency(pull_GBps=, load_GBps=)`, `simulate(..., delay_after_add_s, spot_rate_per_node_hr)` — the lab does not import the core, but check any copied snippets.
- Layer-04 integration must-dos: (1) `serving-engine/PRIMER.md` line ~217 cites the mini engine's `admit_whole_prompt` where it means vLLM — cite `allocate_slots(..., full_sequence_must_fit=scheduler_reserve_full_isl)` (vllm/v1/core/sched/scheduler.py) unless the 04 core review already fixed it; (2) after the 04 lab review lands, re-execute `vllm-internals/notebooks/01_block_hashes_and_eviction.ipynb` — its §4.7/§8.2 asserts import `servelab.sizing` and pin its overhead model; (3) `vllm-internals/source-map.md`: two reading slots are too tight (sitting 3 first slot ~750–900 lines in 50 min; sitting 2 `update_from_output` 445 lines in 30 min) — widen or trim; (4) list `vllm-internals/` and the FA deep dive in `04-inference-engine/README.md`, inject the Colab bootstrap into `flash-attention/flash_attention_deep_dive.ipynb` (`python3 tools/inject_colab_bootstrap.py 04-inference-engine/flash-attention/flash_attention_deep_dive.ipynb`), regenerate the Colab index.
- Layer-04 integration, also: PRIMER §2/§3 credits the 512-token-budget capacity gain to hybrid batching alone; the simulator shows it is partly fewer preemptions (peak KV 38% vs 100%, 0 vs 3–8 preemptions) — add one honest sentence.
