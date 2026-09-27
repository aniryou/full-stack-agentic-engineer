# Structure plan — the five decisions from the 2026-09-26 review, approved 2026-09-26

The owner approved all five items the fix plan (`2026-09-26-fix-plan.md`, "Decisions left to the owner") had held back, with
"use own judgment". This ledger is the resume document for that work: read it first if the session that runs it is cut off.
Same protocol as the fix plan: one branch per package (`claude/clever-hypatia-ma6t1l-<id>`), a draft PR per package, an
adversarial verifier before every merge, squash merges to `main`, workers push after every commit, this ledger is updated on
the tracking branch `claude/clever-hypatia-ma6t1l` after every merge.

## The decisions, and the judgment calls made for them

1. **Consolidate the 07 durable sub-tree** to one primer + one core + one lab. Judgment: keep the `lra` lineage as the review
   proposes (`lra-core` is the strongest engine; `lra-gcp` is pydantic-only), fold in the unique pieces of
   `long-running-agents-core/-gcp` (the limits table and estimates from its primer, drills B1–B8, `adk/nightly_workflow.py`, any
   exercise the lra notebooks lack), keep the Mistral durable lab's genuinely Mistral material (Mistral Workflows / Temporal-style
   `mistral_workflow.py`, `mistral_model.py`, `local_temporal.py`) as an optional provider path of the consolidated lab, and
   remove the wrapper directories (`lra/`, `lra-core/`, `long-running-agentic/`) so the topic reads
   `long-running-durable/{PRIMER.md, lra-core/, lra-gcp/}`. The consolidated lab adopts the standard notebook layout
   (`notebooks/` + `solutions/`) as part of the move.
2. **Fold the three Mistral variants into adapters.** `mistral-agent-core` → `agent-core` with a `mistral` extra, an adapter module,
   `docs/MISTRAL.md`, one provider notebook (`05_going_live_on_mistral`) and its tests; `agentic-scaling-lab-mistral` → a backend
   switch in `agentic-scaling-lab` (`serving.py`, fleet sizing, the vLLM/hybrid backends; the Mistral scaling primer stays as a
   second doc because it is a real rewrite); `agentic-identity-core-mistral` → a `mistral` module of `agentic-identity-core` that
   imports the five moves and keeps the function-calling loop, moderation screener and per-agent keys. The three copies are
   removed; the `scalelab` name clash disappears with them.
3. **One notebook layout**: `notebooks/` holds what a learner opens (exercise blanks, and lessons where a lab has them);
   `solutions/` holds the worked answers, one file per blank with the same name; no `practice/`, `worked/`, `exercises/`,
   nested `notebooks/solutions/`, `_practice`/`_worked`/`_solved` variants. The generators (`tools/gen_colab_index.py`,
   `tools/site/build_site_content.py`) and `tools/ci` then drop their special cases. Runs after 1 and 2 have merged.
4. **Two new topics**, built with `tools/orchestration/build_topic.js` (research → primer + core ∥ lab → nested adversarial review)
   against a new §6b block each in `tools/orchestration/SPEC.md`: `06-gateway/llm-gateway/` (model routing and fallback chains,
   semantic caching, token metering and chargeback, streaming-aware rate limits, virtual keys and tenant isolation, guardrail
   placement, OpenTelemetry GenAI signals, the MCP client-side authorization flow) and `07-application-agent-framework/agent-memory/`
   (memory types, the write path with provenance, retrieval scoring, consolidation and forgetting, the context budget, tenancy and
   deletion, evaluation). No new Terraform: `deploy/gcp` points at existing labs' deploys.
5. **Licence**: keep MIT for everything not otherwise licensed, prose included (the root README already says so); no CC BY split.
   Decision recorded, nothing to change.

## Packages

| id | branch suffix | scope (files owned) | depends on | state |
|---|---|---|---|---|
| `c1-durable` | `-c1-durable` | `07-application-agent-framework/long-running-durable/**`; the durable row and caveats in `07-.../README.md`; CURRICULUM module 07.2 and its drills; COMPUTE.md rows for the durable labs; `tools/ci/labs.json` entries for the durable labs; regenerated Colab links and `mkdocs.yml` | — | planned |
| `c2-mistral` | `-c2-mistral` | `07-.../agent-fundamentals/{agent-core,mistral-agent-core}/**`; `06-gateway/scaling-admission-cost/**`; `06-gateway/identity-security/{agentic-identity-core,agentic-identity-core-mistral}/**` and the 06 topic READMEs; the Mistral mentions in the 06/07 layer READMEs, CURRICULUM and COMPUTE.md; `tools/ci/labs.json`; regenerated files | — | planned |
| `c3-nbdirs` | `-c3-nbdirs` | notebook directory moves outside the two new topics and the consolidated durable lab: 06 identity core + gcp-lab, embeddings-lab, 00 capacity planning, anything else the generators flag; `tools/gen_colab_index.py`, `tools/site/build_site_content.py` and their tests; `tools/ci/labs.json` paths and `ci.py`; `COLAB.md`; the affected READMEs, Makefiles, builders and tooling tests | c1, c2 merged | planned |
| `c4-gateway` | `-c4-gateway` | new `06-gateway/llm-gateway/**`; a row in `06-gateway/README.md`; CURRICULUM module + backlog line; COMPUTE.md §6 rows; `tools/ci/labs.json`; SPEC §6b block and `tools/orchestration/facts/llm-gateway.md` | SPEC block written | planned |
| `c5-memory` | `-c5-memory` | new `07-application-agent-framework/agent-memory/**`; a row in the 07 README; CURRICULUM module + backlog line; COMPUTE.md §6 rows; `tools/ci/labs.json`; SPEC §6b block and `tools/orchestration/facts/agent-memory.md` | SPEC block written | planned |
| `c6-final` | tracking branch | CLAUDE.md (decisions log, percent-source list, baseline), STATUS.md, this ledger's close-out, re-measured counts, the licence decision line | all merged | planned |

Shared files several packages touch (CURRICULUM.md, COMPUTE.md, the 06/07 layer READMEs, `tools/ci/labs.json`, generated
Colab sections and `mkdocs.yml`): edit only your own rows/lines, merge `origin/main` into your branch before the final push,
regenerate rather than hand-merge generated files (`python3 tools/gen_colab_index.py`, `python3 tools/site/build_site_content.py`).

## Resume procedure (if the session is cut off)
1. `git ls-remote --heads origin 'claude/clever-hypatia-ma6t1l-c*'` and the open PRs tell you which packages exist and how far they got.
2. A package whose PR is open and whose verifier passed (see the log below) → merge it (ready → squash, title "<PR title> (#N)").
3. A package with a branch but no verdict → re-launch its workflow from the branch (the fixer prompt reuses an existing worktree/branch).
4. `c3-nbdirs` only after `c1` and `c2` are on `main`; `c6-final` last.

## Log
- 2026-09-26 20:40Z — approval received ("Yes, let's do all of these. Use own judgment."). Ledger opened; scratch orchestration dir
  prepared (SPEC, FACTS, README-STYLE, build_topic.js, review_workflow.js, fixpkg.js, helpers). Next: draft the two §6b blocks,
  launch `c1-durable` and `c2-mistral`, then `c4-gateway` and `c5-memory` once their blocks are in SPEC.md.
- 2026-09-26 23:10Z — tracking PR #35 opened (draft). Launched `c1-durable` (workflow wf_37801607-4e8, script `scratchpad/orch/restructure-c1.js`) and `c2-mistral` (wf_c6959350-46f, `restructure-c2.js`); both scripts are `fixpkg.js` with the wording changed from findings to acceptance criteria, and the package brief embedded. Two drafting agents are writing the SPEC §6b blocks for `llm-gateway` and `agent-memory` (`scratchpad/orch/spec-*.md`). Check-in armed for 00:08Z.
- 2026-09-26 23:50Z — both §6b blocks drafted and reviewed (`llm-gateway` = module 06.7, `agent-memory` = module 07.6; 56 lines each, in SPEC.md's §6b under "Added 2026-09-26 (structure plan)"). Launched `c4-gateway` (workflow wf_a621f736-361) and `c5-memory` (wf_2c633a88-cda) with `build_topic.js` (research → primer + core ∥ lab → nested adversarial review) in worktrees on their branches; a background loop snapshots each worktree to its branch every 20 minutes. Environment differences for these builds (shared venv, no torch, no Terraform) are recorded in FACTS.md's Environment section.
- 2026-09-26 23:50Z — `c2-mistral` worker opened draft PR #36 (branch `claude/clever-hypatia-ma6t1l-c2-mistral`, head 9e48f64; five commits); its verifier is running. Its body lists out-of-scope stale-path hits for the orchestrator (`minengine/perf.py` docstring, `tools/orchestration/facts/rl-and-thinking-models.md`, `tools/site/` comments and fixtures) and a root README notebook count 347 → 333 for `c6-final`.
- 2026-09-27 00:25Z — `c1-durable` VERIFIED (2 rounds; PR #37, head 880c16d + the orchestrator's follow-up commit): one lineage (`long-running-durable/{README.md, PRIMER.md, lra-core/, lra-gcp/}`), the other two folded in (drills, ADK example as an `adk` extra, Mistral Workflows as a `mistral` extra with tests on 3.12 in CI), standard notebook layout, lra-gcp 73 tests (66 pass / 7 skip by default). Round 1 found a real bug (an unknown tool failed the run; now fed back to the model as a ToolError, with tests). The orchestrator applied the fixer's patch to tools/site and gen_colab_index.py (comments, unskipped tests) so the stale-path grep is clean. Left for `c6-final`: CLAUDE.md (builder path line, housekeeping primer paths, decisions-log entry, baseline 339 notebooks before the new topics). Residual: `lra-core/tools/run_notebooks.py --help` treats it as a path (cosmetic). Waiting for CI on the new head, then merge.
- 2026-09-27 00:40Z — `c1-durable` MERGED (PR #37, `d74de20`, squash). Next: when `c2-mistral`'s verdict arrives, merge `main` into its branch (expect conflicts in `tools/ci/tests/test_ci.py` — both packages lowered the builder count 26 → 25, the true count after both is 24 — CURRICULUM.md, COMPUTE.md, the 07 README, `tools/ci/labs.json`; regenerate the Colab sections and `mkdocs.yml`), re-run `run_local.sh --check --notebooks --colab-index --docs`, push, wait for CI, merge; then launch `c3-nbdirs`.
- 2026-09-27 00:55Z — `c2-mistral` workflow ended after 3 rounds with content VERIFIED (rounds 1–2 fixed a `mistralai` 2.x import bug, a vacuous `or True` check, a wrong TPM figure and env-leaking tests) and one open defect: the branch no longer merges (main moved to d74de20 with c1). An agent is merging `origin/main` into the branch, reconciling `test_ci.py` (lab ids; builder count → 24) and `labs.json`, fixing the approved out-of-scope stale references (minengine/perf.py docstring, facts/rl-and-thinking-models.md, the RL primer's line, tools/site example folder names) and neutralising named organisations in the carried Mistral scaling primer. Left for `c6-final`: CLAUDE.md (decisions-log entry, drop the `scalelab` housekeeping bullet, remove `mistral-agent-core` from the percent-source list, baseline 333 → then the new topics), root README notebook count.
- 2026-09-27 01:20Z — `c2-mistral` MERGED (PR #36, `399e4fe`, squash) after the merge agent brought `main` (#37) into the branch: the durable and Mistral changes coexist; `tools/ci/labs.json` has 34 labs, the builder count is 24, `--check/--notebooks/--colab-index/--docs` and the four lab jobs passed locally and CI was green on the final head. Extra edits accepted: the carried Mistral primer and the GCP mapping name no organisation; CURRICULUM §2 says agent fundamentals has 2 labs. Both restructure packages are on `main`; launched `c3-nbdirs` (workflow wf_2f97f7af-bc2, script `scratchpad/orch/restructure-c3.js`). Left for `c6-final`: CLAUDE.md (two decisions-log entries, percent-source list without `mistral-agent-core`, no `scalelab` housekeeping bullet, builder path line, primer paths), root README notebook count (325 before the new topics), `tools/ci/run_local.sh` line 5 ("about 38 venvs" → 34). Residual: `mini-engine-core`'s tooling test `test_expect_fail_tells_an_exercise_stop_from_an_environment_failure` failed once (StopIteration) then passed five times — the same intermittent failure the fix plan saw in thinking-lab.
- 2026-09-27 01:56Z — `c3-nbdirs` worker opened draft PR #38 (head 5455ba6, six commits): 55 deviations in 9 labs moved to `notebooks/` + `solutions/` (cell ids and outputs unchanged), the generators reduced to one rule, a layout guard in `tools/ci/ci.py check` with tests (`tools/ci/nb_layout.py`); its verifier is running. Note for c4/c5 integration: the guard requires the new topics' notebooks to sit in `notebooks/` and `solutions/` with identical file names (the SPEC §3 layout). Left for `c6-final`: CLAUDE.md line naming `00-foundations/transformers/practice/build_notebooks.py` → `tools/build_notebooks.py`.
- 2026-09-27 02:05Z — `c5-memory` BUILT AND REVIEWED (workflow wf_2c633a88-cda): research 145 facts (10 marked unverified: the generative-agents paper form, LongMemEval counts/licence, LoCoMo CC BY-NC, SQLite 3.42 for FTS5 secure-delete, Colab's SQLite, some vLLM flags, managed services); PRIMER.md 668 lines §1–§9; `memcore` 10 modules; `memlab` with SQLite/FTS5 store, pgvector twin, service, agent, consolidation job, fake server; review 32 findings (1 blocking, 15 major, 16 minor), all fixed, none rejected; validator PASS: memory-core 84 passed / 1 skipped, memory-lab 174 passed / 1 skipped, 5/5 solutions and 5/5 blanks in each, rebuild a no-op, 0 broken links. Corrections the research forced on the spec: the generative-agents code's recency is rank-based and favours the oldest-accessed memory (both forms pinned), mem0 2.0 is ADD-only, Graphiti has valid_at/invalid_at (no valid_to). Snapshot loop stopped; an integration agent now merges `main`, links the durable paths, adds labs.json entries, CURRICULUM 07.6, COMPUTE rows, the 07 README row and the facts file, regenerates, validates and opens the draft PR.
