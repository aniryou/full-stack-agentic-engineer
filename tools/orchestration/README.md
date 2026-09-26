# tools/orchestration — how layers 01–05 and the four §6b topics were built and validated

Working notes for the multi-agent build of the layer-01–05 labs and of four more topics — mixture-of-experts, RL and
thinking models, quantization, sandboxed execution (September 2026). Kept in the repo so the build can be resumed
from any checkout, and so the validation steps stay reproducible.

| File | What it is |
|---|---|
| `SPEC.md` | The shared contract every builder and reviewer followed: goal, run tiers T0–T3, fixed paths, conventions, per-layer plan; §6b adds the four later topics. |
| `FACTS.md` | Product facts verified on 2026-09-26 (API versions, metric names, prices, obtainability) — the source of truth for `(verify)` items. |
| `facts/<topic>.md` | The dated fact sheets the research agents wrote for the four §6b topics (`mixture-of-experts`, `rl-and-thinking-models`, `quantization`, `sandboxed-execution`), with the upstream file each fact was read from — the verification record behind those primers. |
| `build_topic.js` | Workflow script for one topic: a research agent writes `facts-<topic>.md`, then builder A (PRIMER + README + core) and builder B (lab) run in parallel, then the nested review. Takes `topic`, `sp`, `repo` and the paths in `args`. |
| `review_workflow.js` | The nested review: three adversarial reviewers (concepts, runnability, pedagogy) → a fixer that verifies each finding before acting → an independent validator that re-runs everything and audits the fixer's rejections. Takes `topic`, `sp`, `repo`, the topic's `dirs` and its SPEC block in `args`. |
| `INTEGRATION.md` | The checklist the integrator followed to land the four §6b topics: layer READMEs, cross-links, `CURRICULUM.md`, `COMPUTE.md`, root README, `CLAUDE.md`, this folder, the site build. |
| `README-STYLE.md` | The README style guide every integrator applies (human-first structure: promise → start here → what you get → run it → how it fits). |
| `STATUS.md` | Per-layer progress: built / reviewed / merged, with the next step. Update it whenever a layer changes state. |
| `tfcheck.sh <dir>` | `terraform fmt -check` + `init` (offline, filesystem provider mirror) + `validate` on a copy of a Terraform directory. Reads `ORCH_SCRATCH` (default `/tmp/orch`, where the mirror lives), `TERRAFORM_BIN` (default `terraform`) and `TF_CLI_CONFIG_FILE` (default `$ORCH_SCRATCH/tf/terraformrc`); set all three as below. |
| `tfattrs.py <resource> [filter...] [--desc]` | Look up Terraform attribute paths from a `terraform providers schema -json` dump (set `TF_SCHEMA_JSON`). |
| `mdlinks.py <paths...>` | Check that relative Markdown links and images resolve. |
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
