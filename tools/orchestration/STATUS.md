# Build status — layers 01–05 (+ deep primers)

Legend: `building` (agents writing) → `built` (builder validation passed) → `reviewed` (adversarial review + fixes passed) → `merged` (on `main`).
WIP snapshots are pushed to `claude/gifted-johnson-9gjwzc` (draft PR). Reviewed layers are merged to `main` through their own branch/PR.

| Layer / item | Paths | State | Next step |
|---|---|---|---|
| 01 roofline-and-fabric | `01-hardware-gpu-fabric/roofline-and-fabric/` (PRIMER, `roofline-core`, `gpu-bench-lab`) | MERGED to main (PR #2, f36a55f) | readability pass rides the -l03 PR |
| 02 cuda-and-nccl | `02-cuda-nccl-runtime/cuda-and-nccl/` (PRIMER, `cuda-nccl-core`, `cuda-nccl-lab`) | MERGED to main (PR #4, 9ca4a04) | — |
| 03 gpu-scheduling | `03-kubernetes-gpu/gpu-scheduling/` (PRIMER, `k8s-gpu-core`, `k8s-gpu-lab`) | MERGED to main (PR #5, 5af4925; incl. README readability pass for root/01/03/05) | — |
| 04 serving-engine | `04-inference-engine/serving-engine/` (PRIMER, `mini-engine-core`, `vllm-serving-lab`) | MERGED to main (PR #6, 916fbef) with vllm-internals + FA deep dive | — |
| 05 serving-orchestration | `05-orchestrator/serving-orchestration/` (PRIMER, `orchestrator-core`, `inference-gateway-lab`) | MERGED to main (PR #3, fb09cfc) | — |
| vLLM internals primer | `04-inference-engine/vllm-internals/` (primer, source-map, 1 notebook) | MERGED to main (PR #6) | — |
| FlashAttention deep dive | `04-inference-engine/flash-attention/` (deep-dive.md, fa_calculators.py + 49 tests, deep_dive notebook; practice notebook repaired) | MERGED to main (PR #6) | — |
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
- DONE in the final pass: root `README.md` and the layer READMEs still describe layer 04's new topics as in progress (01, 02, 03 and 05 are integrated); the integration pass rewrites them and regenerates Colab links.
- `.gitignore` now excludes `terraform.tfvars` / `*.auto.tfvars` (COMPUTE.md tells learners to check with `git check-ignore`).
- vLLM `main` (commit 5840d95, 2026-09-25; PyPI 0.30.0): Model Runner V2 and async scheduling are default-on; `VLLM_USE_V1` and `VLLM_ATTENTION_BACKEND` were removed. The layer-04 lab review must check the lab's env vars/flags against this.
- DONE by the vLLM review validator: §6.4 71.1× like-for-like figure added; §6.3 FP8-KV condition (FA3 on SM90 / FA4) added.
- DONE at the layer-03 integration: no k8s-gpu-lab text asserted it (the kindsim predictor does not model devices); `deploy/gpu-vm/README.md` now says a 1-GPU VM shows one UUID and a multi-GPU VM spreads replicas; no copied core snippets. Was: after the 03 lab review completes, check k8s-gpu-lab (kindsim predictor, notebooks) does not assert that time-sliced replicas share one GPU on a fresh node: the real NVIDIA device plugin's distributed allocation takes replicas from the least-loaded GPUs (fixed in the core: `gpusched/deviceplugin.py`). Core API also changed: `startup_latency(pull_GBps=, load_GBps=)`, `simulate(..., delay_after_add_s, spot_rate_per_node_hr)` — the lab does not import the core, but check any copied snippets.
- Layer 03 went to `main` before layers 02/04/05 and the root docs, so its references to them are plain text (repo paths in backticks), not links: `COMPUTE.md` / `CURRICULUM.md` (topic README, PRIMER §7.2 and §10.3, core and lab READMEs, lab `deploy/gpu-vm/README.md`) and the 02 cuda-and-nccl, 04 serving-engine and 05 serving-orchestration primers (PRIMER §1.2, §4.3, §8, §9; topic README *Builds on*; lab notebook 04). Re-link them when those land (grep `03-kubernetes-gpu` for `COMPUTE.md`, `CURRICULUM.md` and `/PRIMER.md`).
- Layer-04 integration must-dos: (1) `serving-engine/PRIMER.md` line ~217 cites the mini engine's `admit_whole_prompt` where it means vLLM — cite `allocate_slots(..., full_sequence_must_fit=scheduler_reserve_full_isl)` (vllm/v1/core/sched/scheduler.py) unless the 04 core review already fixed it; (2) after the 04 lab review lands, re-execute `vllm-internals/notebooks/01_block_hashes_and_eviction.ipynb` — its §4.7/§8.2 asserts import `servelab.sizing` and pin its overhead model; (3) `vllm-internals/source-map.md`: two reading slots are too tight (sitting 3 first slot ~750–900 lines in 50 min; sitting 2 `update_from_output` 445 lines in 30 min) — widen or trim; (4) list `vllm-internals/` and the FA deep dive in `04-inference-engine/README.md`, inject the Colab bootstrap into `flash-attention/flash_attention_deep_dive.ipynb` (`python3 tools/inject_colab_bootstrap.py 04-inference-engine/flash-attention/flash_attention_deep_dive.ipynb`), regenerate the Colab index.
- Layer-04 integration, also: PRIMER §2/§3 credits the 512-token-budget capacity gain to hybrid batching alone; the simulator shows it is partly fewer preemptions (peak KV 38% vs 100%, 0 vs 3–8 preemptions) — add one honest sentence.
- README readability (user request 2026-09-26): every integrator applies `tools/orchestration/README-STYLE.md` to the layer, topic, core and lab READMEs it merges, and updates the root README row/counts. A readability pass for layers 01+03 and the root README runs on the -l03 branch before its PR; a final root-README rewrite (with CURRICULUM.md/COMPUTE.md links) closes the project.
- Layer-05 integration must check these primer items (deferred by the lab review; the core review may have fixed them): PRIMER §3.2 'higher values are always served first' needs the flow-control-off semantics (llm-d router v0.10.0 default: priority only marks negative-priority objectives sheddable with 429 at saturation); §2.4 the 3:2:2 weights are the llm-d chart default, not 'illustrative'; §2.4 token-load row must match the lab's TokenLoadScorer (this request's uncached tokens); §2.5 add that the router ranking can flip with engine contention and workload.
