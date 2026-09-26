# Integration checklist — four new topics (2026-09-26)

Topics (all built, reviewed and validated on branch `claude/four-new-topics`):
| Layer | Topic dir | Core (pkg) | Lab (pkg) | Module id |
|---|---|---|---|---|
| 00 | `00-foundations/mixture-of-experts/` | `moe-core` (`moecore`) | `moe-lab` (`moelab`) | **00.4** |
| 00 | `00-foundations/rl-and-thinking-models/` | `rl-core` (`rlcore`) | `thinking-lab` (`thinklab`) | **00.5** |
| 04 | `04-inference-engine/quantization/` | `quant-core` (`quantcore`) | `quant-lab` (`quantlab`) | **04.9** |
| 07 | `07-application-agent-framework/sandboxed-execution/` | `sandbox-core` (`sandboxcore`) | `sandbox-lab` (`sandboxlab`) | **07.5** |

Rules: follow `tools/orchestration/README-STYLE.md`; numbers only if computed by code in the repo or marked `(verify)`; keep every
`<!-- colab-links:start/end -->` block untouched (the generator owns it); no emojis; the writing rule from CLAUDE.md. Read each new topic's
README.md, PRIMER.md (section titles, "In a design review", Verify list), core README and lab README before writing a single row about it —
every module row, hour estimate and tier must come from those files (sum the notebooks' stated times). Do not edit anything under the four topic
dirs except to fix a broken link you introduced.

## 1. Layer READMEs (00, 04, 07)
- `00-foundations/README.md`: add both topics; restyle to README-STYLE (title + promise, where the layer sits, a topic table with "You will be
  able to…", time, tier; start here; run it; how it fits; caveats) like `04-inference-engine/README.md`. Keep the existing three topics' rows.
  Update "Covers" and "Signal keywords" (add MoE, router, expert parallelism, RL, RLHF, DPO, GRPO, thinking models, test-time compute).
- `04-inference-engine/README.md`: add a `quantization/` row to the topic table (after `serving-engine/`), the reading order (after serving-engine,
  before or beside vllm-internals), a `Run it` line for `quant-core`, and the scope line.
- `07-application-agent-framework/README.md`: add a `sandboxed-execution/` section under "Current contents" (or restyle to a table like 04), update
  "Covers" and "Signal keywords" (sandbox, gVisor, Firecracker, RuntimeClass, NetworkPolicy, egress proxy, code execution).
- Then `python3 tools/gen_colab_index.py` (rewrites the Colab sections of every layer README and COLAB.md).

## 2. Cross-links from existing material (surgical one-line edits; do not restructure)
- `00-foundations/transformers/docs/transformer-primer.md` §9 MoE row → "worked in depth in [mixture-of-experts](../../mixture-of-experts/PRIMER.md)";
  §6 (training) → one sentence pointing at rl-and-thinking-models §1–4 for post-training.
- `00-foundations/gpu-capacity-planning/PRIMER.md`: the Mistral Large 3 MoE section → link the MoE primer §5/§7; add one line on thinking models
  (long outputs) → rl-and-thinking-models §7.
- `01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md` §3.6 → "the router and the experts themselves: 00-foundations/mixture-of-experts".
- `02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md` §5 expert-parallelism paragraph → link MoE primer §6.
- `04-inference-engine/serving-engine/PRIMER.md` §8 → "deep dive: [quantization](../quantization/PRIMER.md)"; §9 EP → MoE §6; §6/§11 → one line to
  rl-and-thinking-models §7 (thinking workloads). `04-inference-engine/serving-engine/README.md` caveats or "How it fits" → link quantization.
- `04-inference-engine/vllm-internals/vllm-internals-primer.md` §9 → link MoE §6 and rl §7 (reasoning parsers) in one line each.
- `05-orchestrator/serving-orchestration/PRIMER.md` §8 → link MoE §6.
- `06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md` §6.2 → "built out in 07-application-agent-framework/sandboxed-execution".
- `03-kubernetes-gpu/gpu-scheduling/PRIMER.md` §9 or §10 → one line: sandbox pods (RuntimeClass, restricted PSS) are worked in the 07 sandbox topic.
- `07-application-agent-framework/agent-fundamentals/agent-core/README.md` → one line: running a `run_code` tool safely is the sandboxed-execution topic.
Run `python3 tools/orchestration/mdlinks.py` on every file you touched.

## 3. CURRICULUM.md
- §2 table: update the 00, 04 and 07 rows ("What is here" gains the new topics with links; "Not covered yet" loses MoE, quantization depth, sandboxing).
- §3.2 path: insert steps — 00.4 MoE after step 8 (needs the roofline), 04.9 quantization after step 11 (serving-engine §8 measured), 00.5 RL and
  thinking after step 12 (the engine read; thinking workloads reshape 05/06), 07.5 sandboxed execution after step 22 (07.1/07.2). Renumber steps,
  update the totals sentence (recompute the sum; the layers 01–05 subtotal changes only by 04.9). §3.3 routes: add 04.9's core notebooks 01–03 to
  "Serving-infrastructure core", and 07.5's core plus rl §7 to "Agent builder"; update their hours.
- §4: under 00, add two tables in the 01–05 format (Module | You can … | Primer | Core notebook | Lab notebook | Hours | Tier) for 00.4 and 00.5,
  each introduced by the topic's promise line and links to PRIMER/core/lab; under 04 add 04.9 the same way (after 04.8); under 07 add 07.5.
  One row per core notebook (paired with its lab notebook) — read the notebooks' "Tier" lines and titles; hours from the topic README tier table.
- §5.1: add rows for the four primers' "In a design review" (and their core/lab notebook closers). §5.2: add four cross-layer drills that use the
  new modules (thinking workloads vs KV/ITL; INT4 slowing prefill; MoE at small batch vs dense; a code tool exfiltrating a credential), answer
  sketches in the existing style, module lists.
- §6 backlog: remove "MoE architectures" (built); keep attention variants; note nothing else was built.
- Update the header paragraph ("Layers 01–05 each have a topic in the same shape … layer 04 also has two deep dives") to mention that 00, 04 and 07
  now also have primer+core+lab topics; update "Budget about 185 hours" and the one-minute version bullets accordingly.

## 4. COMPUTE.md
- §6 "Which lab needs which tier": add a subsection per new lab (`thinking-lab`, `moe-lab`, `quant-lab`, `sandbox-lab`) in the existing table
  format (Notebook or target | T0 path | Real run | Cheapest real option | GCP (T3)), from each lab README's tier table and deploy READMEs.
  Where a lab points at another lab's GCP deploy, say so in the GCP column. §6's intro line about "00, 06, 07 and the other layer-04 topics"
  must be corrected. §9 Verify list: add the new dated items the labs introduced (model ids, gVisor/GKE Sandbox, llm-compressor versions).

## 5. Root README.md
- "What is inside": update the 00, 04 and 07 rows (topics, "what you will be able to do", paths with links to the new topic READMEs).
- The notebook count wherever it appears (`find . -name '*.ipynb' ! -path '*/.ipynb_checkpoints/*' | wc -l` after the build) and the layout section
  if it lists topics. Do not touch the stack diagram unless a layer's one-line description needs a word.

## 6. CLAUDE.md
- The stack table: add signal keywords (00: MoE, router, expert parallelism, RL, RLHF, DPO, GRPO, thinking, test-time compute; 04: quantization,
  GPTQ, AWQ, FP8, FP4, KV quantization; 07: sandbox, gVisor, code execution). The sentence "layers 01–05 each hold a primer + core + lab topic"
  → now also 00 (two), 04 (quantization) and 07 (sandboxed-execution).
- Decisions log: one dated entry (2026-09-26) in the style of the existing ones: what landed (four topics, packages, notebook and test counts per
  core/lab from the reports, tiers, deploy targets, the §6b SPEC addition, the review workflow), what the primers took at integration, the
  new notebook baseline.

## 7. tools/orchestration
- `STATUS.md`: four new rows (state: reviewed → merged via this branch's PR). `README.md`: mention `build_topic.js` (research → build → nested
  review) and that `review_workflow.js` now takes `sp`/`repo` from args; copy both scripts from the scratchpad. `FACTS.md`: append a pointer to
  `tools/orchestration/facts/<topic>.md` and copy the four fact sheets there (they are the dated verification record for the new primers).

## 8. Orchestrator-only (not the integrator): Colab index, site build (`python3 tools/site/build_site_content.py && mkdocs build --strict`),
full test sweep, notebook count, commit, push, PR, merge, Pages verification.

## 9. Known items from the baseline site build (2026-09-26 12:05, `mkdocs build --strict` passed, 347 notebooks)
- `07-application-agent-framework/sandboxed-execution/sandbox-lab/notebooks_src/05_gke_sandbox_with_gvisor.py` links
  `../../../07-application-agent-framework/sandboxed-execution/sandbox-lab/deploy/gcp/README.md` (resolves outside the repo from
  `notebooks/`); it should be `../deploy/gcp/README.md`. Fix in the source, rebuild the notebooks, re-run the site generator (it reports
  "missing targets" — must be 0).
