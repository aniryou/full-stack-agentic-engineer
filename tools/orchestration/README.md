# tools/orchestration — how layers 01–05, the six §6b topics and the §6c topic were built and validated

Working notes for the multi-agent build of the layer-01–05 labs and of seven more topics — mixture-of-experts, RL and
thinking models, quantization, sandboxed execution, and, in the 2026-09-26 structure plan, the LLM gateway and agent memory
(§6b), and distillation (§6c) (September 2026). Kept in the repo so the build can be resumed from any checkout, and so the
validation steps stay reproducible.

| File | What it is |
|---|---|
| `SPEC.md` | The shared contract every builder and reviewer followed: goal, run tiers T0–T3, fixed paths, conventions, per-layer plan; §6b adds the six later topics (four, then two in the structure plan), §6c distillation. |
| `FACTS.md` | Product facts verified on 2026-09-26 (API versions, metric names, prices, obtainability) — the source of truth for `(verify)` items. |
| `facts/<topic>.md` | The dated fact sheets the research agents wrote for the six §6b topics (`mixture-of-experts`, `rl-and-thinking-models`, `quantization`, `sandboxed-execution`, `llm-gateway`, `agent-memory`) and the §6c topic (`distillation`), with the upstream file each fact was read from — the verification record behind those primers. |
| `build_topic.js` | Workflow script for one topic: a research agent writes `facts-<topic>.md`, then builder A (PRIMER + README + core) and builder B (lab) run in parallel, then the nested review. Takes `topic`, `sp`, `repo` and the paths in `args`. |
| `review_workflow.js` | The nested review: three adversarial reviewers (concepts, runnability, pedagogy) → a fixer that verifies each finding before acting → an independent validator that re-runs everything and audits the fixer's rejections. Takes `topic`, `sp`, `repo`, the topic's `dirs` and its SPEC block in `args`. |
| `INTEGRATION.md` | The checklist the integrator followed to land the four §6b topics: layer READMEs, cross-links, `CURRICULUM.md`, `COMPUTE.md`, root README, `CLAUDE.md`, this folder, the site build; the §6c topic landed with an adapted copy. |
| `README-STYLE.md` | The README style guide every integrator applies (human-first structure: promise → start here → what you get → run it → how it fits). |
| `STATUS.md` | Per-layer progress: built / reviewed / merged, with the next step. Update it whenever a layer changes state. |
| `reviews/` | The 2026-09-26 adversarial review of the whole repo (`2026-09-26-adversarial-review.md`); its fix plan and ledger (`2026-09-26-fix-plan.md`) with `fixpkg.js`, the fix → verify → re-fix workflow each fix package ran; the structure plan that carried out the five decisions the fix plan left to the owner (`2026-09-26-structure-plan.md`); and `structure-plan/`, the scripts and briefs its packages ran with. |
| `tfcheck.sh <dir>` | `terraform fmt -check` + `init` (offline, filesystem provider mirror) + `validate` on a copy of a Terraform directory. Reads `ORCH_SCRATCH` (default `/tmp/orch`, where the mirror lives), `TERRAFORM_BIN` (default `terraform`) and `TF_CLI_CONFIG_FILE` (default `$ORCH_SCRATCH/tf/terraformrc`); set all three as below. |
| `tfattrs.py <resource> [filter...] [--desc]` | Look up Terraform attribute paths from a `terraform providers schema -json` dump (set `TF_SCHEMA_JSON`). |
| `mdlinks.py <paths...>` | Check that relative Markdown links and images resolve. |
| `STE100-STYLE.md` | The ASD-STE100 (Simplified Technical English) brief for the layers' prose: what does not change, the writing rules, the vocabulary (the usual replacements, the second-pass dictionary-level words of 2026-10-30 and the rule on parentheses), the technical names per layer, the document types. |
| `ste_lint.py <paths...>` | The heuristic STE linter over Markdown, notebooks and percent-format sources: sentence and paragraph length, modals, contractions, dashes, arrows, -ing forms, passives, semicolons, the word lists, a clause with a verb inside parentheses (`paren-clause`); `--json` adds the share of long words. |
| `ste_rewrite_workflow.js` | The first STE pass, one layer per run: a rewriter per disjoint file group, an adversarial verifier against `git show HEAD:`, a fixer and a re-check, up to two rounds. |
| `ste_pass2_workflow.js` | The second STE pass (2026-10-30), one layer per run: an agent applies the dictionary-level findings and reads for plain language, a verifier reads the diff hunk by hunk against `HEAD` (or `args.baseline`), a fixer repairs, a re-check follows. |
| `nb_md.py`, `nb_outputs.py`, `nb_exec.py` | Notebook helpers for the prose passes: dump and apply the Markdown cells of a notebook (and its solution twin), carry recorded outputs over to a rebuilt notebook whose code did not change, execute a solution in place in the repository's JSON layout. |
| `FIGURE-STYLE.md` | The brief for figures in the primers: when a figure earns its place, Mermaid or SVG, the Mermaid rules with the one-screen rule (shown in an 800 px column: at most 600 px tall, scaled by no more than 0.7), the SVG house style, placement and captions, the checks. |
| `render_figures.js` | The writer's renderer (headless Chromium, prerequisites in its header): `md <doc> <outdir>` renders every Mermaid block with a one-screen verdict, `svg <file> <png>` one SVG, `check` every figure of the repository into `figures-manifest.tsv`, which `tools/ci/tests/test_figures_fit.py` reads in CI. |
| `figures-manifest.tsv` | Every figure measured by the check mode: kind, document, block, line, the sha1 of the Mermaid source, size, height when shown, scale, verdict. Regenerate it after a figure changes and commit it with the document. |
| `figure_fit_workflow.js` | The one-screen figure workflow (2026-10-30): a reshaper per document makes every block fit and adds compact figures to named sections, a verifier renders and reads every figure against `git show HEAD:`, a fixer repairs, up to two rounds. |
| `svgchart.py` | The chart helper of the house style (2026-10-30): a core's `tools/make_figures.py` builds a spec (axes, series, marks, a note), `render()` draws it as a dark-card SVG with the spec in its `<desc>`, and `check_figure()` in the core's `tests/test_figures.py` pins the committed file to what the core computes now (FIGURE-STYLE.md §7). `python3 tools/orchestration/svgchart.py demo <out.svg>` draws a demonstration chart. |
| `figure_improve_workflow.js` | The figure improvement workflow (2026-10-30, round 3): per document, a chart stage for the topics with a core (the generator, the test and two to four pinned charts in the primer), then an improve stage that judges every figure against the reader's checklist and adds figures to long sections; each stage has an adversarial verifier against a snapshot and a fixer, up to two rounds. |
| `figure_notebooks_workflow.js` | Figures in the notebooks (2026-10-30, FIGURE-STYLE.md §8): per lab, an agent gives each notebook at most one figure a primer already carries, embedded by its absolute URL on `main` in the first Markdown cell with a caption that names the source (through the percent source, the builder or `nb_md.py`), an adversarial verifier checks the lesson, the twins, the builder no-op and the tests, a fixer, up to two rounds. |
| `pipi <pip args>` | `pip install` serialised through a lock file, for several agents sharing one environment. |

## Reproducing the offline Terraform check

Provider downloads from `registry.terraform.io` may be blocked in a sandbox; `releases.hashicorp.com` usually is not.

```bash
export ORCH_SCRATCH=/tmp/orch && mkdir -p $ORCH_SCRATCH/tf/bin $ORCH_SCRATCH/tf/mirror && cd $ORCH_SCRATCH/tf
curl -sSO https://releases.hashicorp.com/terraform/1.13.3/terraform_1.13.3_linux_amd64.zip && python3 -c "import zipfile; zipfile.ZipFile('terraform_1.13.3_linux_amd64.zip').extractall('bin')" && chmod +x bin/terraform
for spec in "google 8.4.0" "google-beta 8.4.0" "kubernetes 3.2.1" "helm 3.3.0" "random 3.7.2" "null 3.2.4"; do set -- $spec
  d=mirror/registry.terraform.io/hashicorp/$1; mkdir -p $d
  curl -sS -o $d/terraform-provider-$1_$2_linux_amd64.zip https://releases.hashicorp.com/terraform-provider-$1/$2/terraform-provider-$1_$2_linux_amd64.zip; done
printf 'provider_installation {\n  filesystem_mirror {\n    path    = "%s/tf/mirror"\n    include = ["registry.terraform.io/*/*"]\n  }\n}\n' "$ORCH_SCRATCH" > terraformrc
export TF_CLI_CONFIG_FILE=$ORCH_SCRATCH/tf/terraformrc TERRAFORM_BIN=$ORCH_SCRATCH/tf/bin/terraform
tools/orchestration/tfcheck.sh 02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/gcp/terraform
```

Kubernetes manifests: `pip install kubernetes-validate && kubernetes-validate --strict -k 1.34.0 <files>` (core kinds; CRDs are
checked by each lab's tests).
