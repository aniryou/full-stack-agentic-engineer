# structure-plan — the scripts and briefs the 2026-09-26 structure plan ran with

Kept so each package of the [structure plan](../2026-09-26-structure-plan.md) can be re-read or re-run. Copied from the
orchestrator's scratch directory as run, except: absolute paths are `$SP` (the scratch directory) and `$REPO` (the
checkout); the model alias and the attribution lines are `$MODEL`, `$COMMIT_TRAILER` and `$PR_TRAILER`; and the curriculum
draft's repo-root links in `spec-llm-gateway.md` are re-rooted to resolve from this folder.

| File | What it is |
|---|---|
| `restructure.js` | Workflow script for one package: a worker in its own worktree and venv commits to the package branch, pushes and opens a draft PR → an independent adversarial verifier runs every acceptance criterion → up to two re-work rounds. It is [`fixpkg.js`](../fixpkg.js) with review findings replaced by acceptance criteria. |
| `args-c1.json`, `args-c2.json`, `args-c3.json` | The briefs of `c1-durable`, `c2-mistral` and `c3-nbdirs`: `pkg` (id, title, branch, scope, the decision and its acceptance criteria, verification commands, notes, allowed exceptions) and `common` (paths, model and effort, the files no package may touch, trailers). |
| `spec-llm-gateway.md`, `spec-agent-memory.md` | The inputs of `c4-gateway` and `c5-memory`: the SPEC §6b block (now in [`SPEC.md`](../../SPEC.md)), the research brief, the existing material to cite, the curriculum draft and, for memory, notes for the integrator. |

How a package ran. `c1`–`c3`: the Workflow tool ran `restructure-<id>.js`, which is `restructure.js` with its
`args-<id>.json` embedded as `const EMBEDDED = {…}` in place of `args`. `c4` and `c5`: [`build_topic.js`](../../build_topic.js)
(research → primer + core ∥ lab → the nested [`review_workflow.js`](../../review_workflow.js)) in a worktree on the package
branch, with `specBlock`, `researchBrief` and `existing` taken from the spec file and the directories and packages from
SPEC §2; both scripts ran with their environment paragraph changed for these builds (one shared venv, no torch, no Terraform,
agents never run git; [`FACTS.md`](../../FACTS.md), "Environment for the 2026-09-26 structure-plan builds"). The workflow
ids, verdicts and merges are in the ledger's log.
