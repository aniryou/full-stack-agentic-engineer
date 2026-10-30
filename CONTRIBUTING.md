# Contributing to full-stack-agentic-engineer

Read this before your first change: after it you know where a change belongs, which conventions it follows, which
checks it must pass and how it lands on `main`.

This is a public learning repository for the LLM serving stack, from the GPUs and fabric at the bottom to agents and
applications at the top, organised as eight layer folders with topic folders inside them. Topics teach through
primers with worked numbers, small from-scratch implementations with fill-in notebooks, and fuller labs with real
tools and deploy recipes. Every concept must be learnable on a laptop or in Colab for free, CI runs every lab's
tests, and in the main topics those tests recompute the numbers the primers quote.

Two kinds of contributor work here: people, and coding agents. [`CLAUDE.md`](CLAUDE.md) is the agent operating
manual: the layer rules, the `raw/` inbox procedure, the notebook and CI mechanics in detail, and the decisions log.
This document is its human-facing counterpart: the layout, the conventions, the checks and the process in one place,
each with a pointer to where it is defined or enforced. Both documents apply to both kinds of contributor. Where this
page and the file a pointer names disagree, the file wins, and this page needs a fix.

## Layout

### Eight layers

The repository mirrors the stack bottom-up, with `00-foundations` below it. Each layer has exactly one name, used in
its README's title, the root README, the curriculum and the site. The names are defined once, in `LAYER_NAMES` in
[`tools/gen_colab_index.py`](tools/gen_colab_index.py); a site test keeps `LAYER_TITLES` in
[`tools/site/build_site_content.py`](tools/site/build_site_content.py) identical.

| Folder | Name | What belongs there |
|---|---|---|
| `00-foundations` | 00 · Foundations | the model itself: transformer internals, capacity planning, the model landscape, mixture-of-experts, RL and thinking models, distillation |
| `01-hardware-gpu-fabric` | 01 · Hardware and fabric | GPUs and their memory, NVLink and NVSwitch, NICs, storage, power and cooling |
| `02-cuda-nccl-runtime` | 02 · CUDA, NCCL and runtime | the driver, CUDA, NCCL collectives, the container runtime, MIG |
| `03-kubernetes-gpu` | 03 · Kubernetes and GPU scheduling | the GPU Operator, device plugins, GPU scheduling, gang and topology-aware placement |
| `04-inference-engine` | 04 · Inference engine | vLLM, SGLang, TensorRT-LLM: the KV cache, batching, attention kernels, quantization |
| `05-orchestrator` | 05 · Orchestrator | Dynamo, llm-d, Ray Serve: routing, autoscaling, prefill/decode disaggregation |
| `06-gateway` | 06 · Gateway | auth, rate limits, routing, quotas, observability, cost |
| `07-application-agent-framework` | 07 · Agents and applications | the agent loop, tools, sandboxed execution, memory, durable execution, RAG, evals |

To place new material, pick one primary layer by the signal keywords in CLAUDE.md's stack table. If it also covers a
second layer, keep it whole and add a one-line cross-reference to both layer READMEs.

### Topics, and the primer + core + lab shape

Inside a layer, material is grouped in topic folders with kebab-case names, such as
`04-inference-engine/serving-engine/`; a lab always moves as a whole unit. The main topics share one shape, and a
new topic uses it:

| Path | What it is | Tier |
|---|---|---|
| `<topic>/README.md` | the index: the order to work the topic, with the time and tier of each step | — |
| `<topic>/PRIMER.md` | the concepts in numbered sections; every formula has a worked number and names the core function that computes it | reading |
| `<topic>/*-core/` | a minimal implementation on the standard library (plus numpy), offline and readable in a sitting, with its notebooks | T0 |
| `<topic>/*-lab/` | the detailed version: T0 fallbacks, GPU code paths and `deploy/<target>/` recipes (a GPU box, kind or compose, Google Cloud) | T0 → T3 |

Older topics vary (a primer with written exercises, a primer plus practice notebooks, a lab of its own), and each
README says what it has. A lab may keep contributor notes of its own, such as
[`gcp-agent-platform-lab/docs/CONTRIBUTING.md`](07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/docs/CONTRIBUTING.md);
they add to this page.

### Files that span layers

| Path | What it is |
|---|---|
| [`CURRICULUM.md`](CURRICULUM.md) | the learning path: modules, order, hours, tiers, design drills, the backlog |
| [`COMPUTE.md`](COMPUTE.md) | where each tier runs, what it costs, which lab needs which tier, a dated verify list |
| [`COLAB.md`](COLAB.md) | running the notebooks in Colab (generated) |
| [`LICENSE`](LICENSE) | the MIT licence for everything without a licence of its own |
| `site/`, `mkdocs.yml` | the guide site: a few hand-written pages, plus pages generated from the layer folders and the three guides above ([`tools/site/README.md`](tools/site/README.md)) |
| `tools/` | the Colab cell injector and link generator; `tools/site/`, the site generator; `tools/ci/`, everything CI runs |
| [`tools/orchestration/`](tools/orchestration/README.md) | how topics are specified, built and reviewed: the spec, the facts, the status, the review ledgers (maintainer material, not lessons) |
| `.github/workflows/` | `tests.yml` (CI) and `pages.yml` (publishes the site from `main`) |
| [`raw/`](raw/README.md) | the inbox for new material |

`raw/` is an inbox: new material is dropped there, then **moved**, never copied, into the layer and topic it belongs
to. Everything in it except its README is gitignored, so nothing there is committed until it has been sorted.
[`raw/README.md`](raw/README.md) says what a good drop looks like, and CLAUDE.md's reorg checklist is the procedure.

## Conventions

SPEC below is [`tools/orchestration/SPEC.md`](tools/orchestration/SPEC.md), the contract new topics are built to.

### Writing

- Address an engineer explaining a design in a design review: never a role, an employer or a customer. A
  provider's stack is the same concept worked on that provider. The recurring callouts are "The one-minute
  version", "In a design review" and design drills. *Defined in* CLAUDE.md's decisions log (2026-09-20) and SPEC §0;
  checked in review.
- Product facts (versions, prices, SKUs, availability) carry `(verify)` and a date, and a primer ends with a dated
  Verify list (SPEC §5). The record behind them is [`tools/orchestration/FACTS.md`](tools/orchestration/FACTS.md)
  and the research fact sheets in [`tools/orchestration/facts/`](tools/orchestration/facts/).
- No emojis, except ✅ in the output of a check, and no marketing adjectives
  ([`README-STYLE.md`](tools/orchestration/README-STYLE.md), SPEC §3).
- One name per layer: the one in the table above, in prose, headings and tables alike.
- Mathematics is TeX, written the way GitHub renders it: `$...$` inline, `$$` on its own lines for display (an
  `aligned` environment inside it for a derivation). No space just inside the dollars, no digit right after the
  closing one, `\|` rather than `|` inside a table cell; a dollar price is never inside math. The site turns the
  inline form into `\( \)` and leaves prices alone (`tools/site/README.md`, "Math"). Plain-text formulas
  (`α·T²·KL(p_T ‖ q_T)`, `r_t = log π_T − log π_S`) render as raw underscores: do not write them.
- Paragraphs carry one idea and stay under about 150 words; a summary of several bold terms is a list, one item
  per term. Callouts are blockquotes with a bold lead-in (`> **Pitfall.**`, `> **Verify.**`, `> **In a design
  review**`) or GitHub alerts (`> [!NOTE]`, `[!TIP]`, `[!IMPORTANT]`, `[!WARNING]`, `[!CAUTION]`), which the site
  renders as admonitions.
- The prose of every layer (`00-foundations` to `07-application-agent-framework`: primers, READMEs, docs and the
  Markdown cells of the notebooks) is in ASD-STE100 (Simplified Technical English), with
  [`tools/orchestration/STE100-STYLE.md`](tools/orchestration/STE100-STYLE.md) as the brief and
  `tools/orchestration/ste_lint.py` as the checker (one topic on 2026-10-03, the rest on 2026-10-04). A change
  there keeps that style; the root documents (this file, `README.md`, `CURRICULUM.md`, `COMPUTE.md`, `COLAB.md`)
  keep the rules above.
- Figures: a mechanism the prose explains over several paragraphs gets a diagram, by
  [`tools/orchestration/FIGURE-STYLE.md`](tools/orchestration/FIGURE-STYLE.md): a Mermaid block (flows, sequences,
  state machines, architectures; no init directive or `classDef`, so GitHub and the site theme it) or a hand-written
  SVG in the house style (a dark card, one teal accent) in `figures/` beside the document (layouts, timelines, memory
  and bit layouts, charts). A figure is an insertion with an italic caption (in STE in the layers), states no fact the
  prose does not, and carries a number only verbatim from the document; the ASCII sketches in code fences stay.
  `node tools/orchestration/render_figures.js md <doc> <outdir>` renders every Mermaid block and SVG to PNG to look at
  (its header names the prerequisites; CI does not run it).

### Run tiers

| Tier | Where | What it promises |
|---|---|---|
| **T0** | a laptop, Colab CPU or CI; free | every concept is learnable here, offline |
| **T1** | one small GPU: a free Colab or Kaggle T4, or a rented card | real kernels, a real engine with a small model, real metrics |
| **T2** | a multi-GPU box, rented for about an hour (or Kaggle's free 2×T4) | collectives, P2P and NVLink, tensor parallelism |
| **T3** | Google Cloud via Terraform, with the cheapest defaults (L4, Spot, zonal, scale-to-zero) | how it looks in production on one cloud |

Every notebook and README step declares its tier. Code at T1 and above detects the GPU, Docker, cluster or cloud it
needs; when that is absent it runs a clearly labelled T0 path and prints what to run on real hardware. Never
fabricate a measurement: simulator output is labelled "simulated", and bundled tool output "sample output in the
documented format (illustrative)". Google Cloud is one deployment target, never a prerequisite. *Defined in* SPEC §1,
and presented for learners in [`CURRICULUM.md` §1.2](CURRICULUM.md#12-run-tiers) and
[`COMPUTE.md` §1](COMPUTE.md#1-run-tiers), with prices. CI runs T0 only: `tools/ci/ci.py run` fails any install
that makes torch importable.

### Numbers

- A number in a primer is either computed by a function in the topic's core, which the primer names (for example
  `roofline.roofline.attainable()`), or marked `(verify)`. The core's tests pin it. The pattern to copy is
  `roofline-core`'s `tests/test_primer_numbers.py`: it recomputes every computed number the primer quotes and
  requires it verbatim, so a changed formula fails until the primer follows.
- A trained or sampled number (a seeded run of a tiny model, a sample drawn from one) is one CPU's run: numpy's
  OpenBLAS picks its matrix kernel per microarchitecture, the kernels round differently in the last bit, and a
  few hundred Adam steps can grow that into a slightly different model. Hold such a number to the primer's value
  within a tolerance measured across CPU variants, never with exact equality, and assert the primer's qualitative
  claim outright; the pattern is `distill-core`'s `near()` in `tests/test_primer_numbers.py` (the primer's text
  verbatim, each `#` a number within its tolerance) with `tools/host_sensitivity.py`, which recomputes the
  numbers under every OpenBLAS kernel and numpy SIMD level the machine can force, plus last-bit perturbations,
  and prints the spread.
- When a formula or an explanation already has a home in the repo, cite it and reuse it; duplicating it is a major
  review finding (SPEC §6b and §6c). Labs are standalone packages (no lab's package imports another lab), so new code that
  needs such a formula re-implements it, and a test reproduces the home's numbers and says so
  (`tests/test_repo_numbers.py`).

### Notebooks

- **One layout**, guarded by [`tools/ci/nb_layout.py`](tools/ci/nb_layout.py) through `tools/ci/ci.py check`:
  `<lab>/notebooks/<name>.ipynb` is what a learner opens (exercise blanks, lessons, walkthroughs), and
  `<lab>/solutions/<name>.ipynb` is the worked answer to the blank with the same file name. There is no other
  notebook folder, no notebook at a lab's top level and no answer key beside its blank under a suffix.
- **Percent-source labs** keep their notebooks as `notebooks_src/NN_name.py` (cells marked `# %% [markdown]`, `# %%`,
  `# %% exercise` with `### BEGIN SOLUTION` / `### END SOLUTION`, and `# %% check`) and build them with
  `python3 tools/build_notebooks.py` in the lab. The builder writes both folders, puts the Colab bootstrap cell (its
  `BOOTSTRAP` constant) first and mints stable cell ids, so rebuilding a clean tree is a no-op. Never edit these
  `.ipynb` files by hand; `python3 tools/ci/ci.py builders` lists every builder. A heading (`# ## …`) needs a
  `# %% [markdown]` line of its own: after a code, exercise or check cell without one, the builder writes it and the
  prose under it as comments at the end of that cell. [`tools/ci/nb_sources.py`](tools/ci/nb_sources.py) checks
  every source for this through `tools/ci/ci.py check`.
- **Hand-written notebooks** get their Colab cell from `python3 tools/inject_colab_bootstrap.py <lab-dir>`, which is
  idempotent and replaces only its own cell. Never run it on a percent-source lab.
- **Exercises:** a blank stops at its first exercise with `NotImplementedError`, and every exercise is followed by a
  check that prints ✅ and must fail a wrong or empty answer; reviewers try both. Blanks are committed blank, so
  `git restore` a notebook you worked in, or rebuild it.
- **Two runs** prove all of this: `python3 tools/run_notebooks.py solutions` (every solution runs clean) and
  `python3 tools/run_notebooks.py notebooks --expect-fail` (every blank stops at an exercise; a missing module or an
  error before the first exercise is reported as `FAIL(env)`). They are each lab's `solutions` step in
  `tools/ci/labs.json`, which CI's manual solutions job runs, and `make check` runs them too. Every lab with a
  `tools/build_notebooks.py` also pins its tooling in `tests/test_notebook_tooling.py`.
- A percent-source notebook opens with a title, a `**Tier:**` line and "The one-minute version", works examples, sets
  three to six exercises each followed by a check, and ends with "In a design review" (SPEC §3).

### Packaging a lab

Each lab is its own package with its own environment; there is no repo-wide one. The details are in SPEC §3.

| File | What it holds |
|---|---|
| `pyproject.toml`, `requirements.txt` | setuptools with the package directory at the lab root (`<pkg>/`); a `dev` extra with pytest and what the notebook runner needs; anything heavy (torch, an engine, a cloud SDK, a provider client) in an optional extra of its own, imported lazily and skipped when absent |
| `Makefile` | `setup`, `test`, `notebooks`, `check`, `lab`, `clean` |
| `tests/` | pytest, offline and quick (under about a minute): no network, no model downloads, no GPU, no keys; one focused test per concept |
| `LICENSE`, `README.md`, `.gitignore` | MIT for a new lab; the README follows the style guide below |
| `deploy/<target>/` | labs only: one folder per target, each with a README that says what it does, what it costs and how to clean up |

No two labs declare the same package name; `tools/ci/tests/test_ci.py` checks it. Terraform lives in
`deploy/gcp/terraform/` with the cheapest defaults and a `terraform.tfvars.example`; `terraform.tfvars`,
`*.auto.tfvars` and state are gitignored. Deploy scripts use `set -euo pipefail` and support `DRY_RUN=1`.

### READMEs

Every README (root, layer, topic, core, lab) follows
[`tools/orchestration/README-STYLE.md`](tools/orchestration/README-STYLE.md): a title with a one-sentence promise →
start here → what you get (path, outcome, time, tier) → run it → how it fits → caveats.
[`06-gateway/README.md`](06-gateway/README.md) is a layer README to copy, and
[`agent-core/README.md`](07-application-agent-framework/agent-fundamentals/agent-core/README.md) a lab README. Counts
and timings are measured, and re-measured in the change that moves them: `python3 -m pytest -q` in the lab for tests
and seconds, `find . -name '*.ipynb' ! -path '*/.ipynb_checkpoints/*' | wc -l` at the root for notebooks; hours
come from `CURRICULUM.md`. A layer README's "Run in Colab" section, between its `colab-links` markers, belongs to
the generator.

### Licences

Everything without a licence of its own (the primers, the curriculum, the compute guide, the site's text and the code
outside the labs) is under the MIT licence in the root [`LICENSE`](LICENSE), prose included. A lab with its own
`LICENSE` keeps it; the root README's [Licence](README.md#licence) section names the exceptions. A new lab ships an
MIT `LICENSE`, and a change is published under the licence of the files it touches.

## Making a change

1. **Branch from an up-to-date `main`** (fork first if you cannot push here): one topic or package per branch and PR.
2. **Change the source, not the output:** a `notebooks_src/*.py` rather than its notebooks, a generator's input
   rather than what it writes.
3. **Run the checks** that cover what you touched, and **regenerate** what is generated.
4. **Update the bookkeeping** the change affects.
5. **Commit** in small, logical steps, each message saying what changed and why.
6. **Push and open a draft pull request**; mark it ready once CI is green. Pull requests are squash-merged into `main`.

### The checks

[`.github/workflows/tests.yml`](.github/workflows/tests.yml) runs on every pull request and every push to `main`.
[`tools/ci/run_local.sh`](tools/ci/run_local.sh) runs the same jobs on a laptop, each in a fresh virtual environment
(it needs `python3.11` on `PATH`, and `python3.12` for the lab entries that ask for it):

| `tools/ci/run_local.sh` | What it checks | CI job |
|---|---|---|
| `--check` | every test file belongs to a lab in `tools/ci/labs.json`, every listed directory exists, every lab with a notebook runner has a solutions step, every notebook follows the layout, every heading in a percent-format notebook source is in a markdown cell; then the tests of `tools/ci` itself | lab list complete, CI helpers tested |
| `<lab-id> …` | the lab installs as its README says, without torch, and its tests pass (`--list` prints the ids) | tests (`<lab-id>`) |
| `--solutions <lab-id>` | every solution notebook runs clean and every blank stops at its first exercise | solutions (manual: Actions → tests → Run workflow) |
| `--notebooks` | every notebook has a Colab setup cell, and every builder and the injector leave the committed tree unchanged (commit first: it needs a clean tree) | notebook rebuilds are no-ops |
| `--colab-index` | `tools/gen_colab_index.py` changes nothing | Colab links up to date |
| `--docs` | the site generator's tests, `mkdocs.yml` regenerated unchanged, a strict site build, and every relative link in every tracked Markdown file | site and links |

Before you push, run `--check` and the labs you touched; add `--notebooks`, `--colab-index` and `--docs` when you
touched notebooks or Markdown. While writing, `python3 tools/orchestration/mdlinks.py <files>` checks links in a
second.

### Generated files

Never edit these by hand; regenerate them, also after a merge that conflicts in them:

| File | Regenerate with |
|---|---|
| a percent-source lab's `notebooks/` and `solutions/` | `python3 tools/build_notebooks.py` in the lab |
| the Colab cell of a hand-written notebook | `python3 tools/inject_colab_bootstrap.py <lab-dir>` |
| [`COLAB.md`](COLAB.md) and the "Run in Colab" section of each layer README | `python3 tools/gen_colab_index.py` |
| the `nav:` and `not_in_nav:` blocks of `mkdocs.yml` | `python3 tools/site/build_site_content.py` |

### Bookkeeping

- **A new lab with tests** needs an entry in [`tools/ci/labs.json`](tools/ci/labs.json), or `--check` fails. Its
  `install` is the line the lab's README gives, minus anything that pulls in torch, and `solutions` is present when
  the lab has a notebook runner:

  ```json
  {"id": "moe-core", "dir": "00-foundations/mixture-of-experts/moe-core", "python": "3.11",
   "install": "python -m pip install -r requirements.txt", "test": "python -m pytest -q",
   "solutions": "python tools/run_notebooks.py solutions && python tools/run_notebooks.py notebooks --expect-fail"}
  ```

- **A new topic** needs its row in the layer README; a module in `CURRICULUM.md` (number, hours and design drills);
  its lab's table in `COMPUTE.md` §6 and its dated items in the §9 Verify list; its entry in the root README; and a
  dated entry in CLAUDE.md's decisions log. [`tools/orchestration/INTEGRATION.md`](tools/orchestration/INTEGRATION.md)
  is the checklist used when the SPEC §6b topics landed, and, adapted, when the §6c topic (distillation) did.
- **Notebooks added or removed:** re-measure the notebook count the root README quotes.
- **A moved or renamed path:** search the whole tree for the old one. READMEs, `CURRICULUM.md`, `COMPUTE.md`,
  `tools/ci/labs.json`, Makefiles, primers and the tooling tests all name paths.

### Pull requests

Open pull requests as drafts. They are squash-merged, so the title becomes the commit on `main`: say what changed,
in plain words. The body carries:

- **What changed and why:** the problem, the change, and anything deliberately left out.
- **Verification:** the exact commands you ran, each with its result (the summary line is enough).
- **Not validated:** what you could not run (a GPU path, Docker, a cloud deployment), and why.
- **`(verify)` items:** every dated product fact the change introduces, so a reviewer can re-check it.

## Adding a whole topic

A new topic in the primer + core + lab shape has a contract:
[`tools/orchestration/SPEC.md`](tools/orchestration/SPEC.md) §0–§5 (goal, tiers, layout, conventions, validation,
the primer contract) plus a block of its own in the style of §6, §6b and §6c, fixing the primer's numbered sections, the
core's modules and notebooks, the lab's notebooks and deploy targets, and the existing material it must cite and
reproduce. Write that block, and have it reviewed, before building.

Facts come before prose: a dated research fact sheet in [`tools/orchestration/facts/`](tools/orchestration/facts/)
records where each product fact was read, and anything in neither it nor `FACTS.md` is `(verify)`. The built topic
then gets an adversarial review through three lenses (concepts, runnability, pedagogy), followed by a fixer that
verifies each finding before acting on it and an independent validator that re-runs everything
([`review_workflow.js`](tools/orchestration/review_workflow.js);
[`build_topic.js`](tools/orchestration/build_topic.js) chains research, build and review for one topic). Track its
state in [`STATUS.md`](tools/orchestration/STATUS.md) and land it with the bookkeeping above. The scripts are the
process written down; a person can follow the same steps by hand.

## For coding agents

Read [`CLAUDE.md`](CLAUDE.md) first: it holds the operating detail this page summarises. Then:

- **One package per branch.** When several agents work at once, give each a disjoint set of files; in shared files
  (the curriculum, the compute guide, layer READMEs, `tools/ci/labs.json`) each edits only its own rows.
- **Push after every commit**, so an interrupted session loses nothing. Bring `main` in with a merge; never rebase
  or force-push a branch that has been pushed.
- **An independent adversarial verifier**, a second agent and never the author, checks every package before it
  merges.
- **Keep a ledger** under `tools/orchestration/reviews/` for any effort longer than one sitting: packages, branches,
  states, a resume procedure and a dated log.
  [`2026-09-26-fix-plan.md`](tools/orchestration/reviews/2026-09-26-fix-plan.md) is the model; the structure plan
  that followed it uses the same format.
- **Never** weaken, skip or delete a test to get green: fix the cause, or report it.
- **Never** hand-edit a generated file (the table above).
- **Never** commit secrets or keys, a real `terraform.tfvars`, model weights, or anything a T0 path would need the
  network to fetch.

## Reporting a problem

Open a [GitHub issue](https://github.com/aniryou/full-stack-agentic-engineer/issues); for a sweep with many findings,
write a review document under `tools/orchestration/reviews/` in the format of
[the 2026-09-26 review](tools/orchestration/reviews/2026-09-26-adversarial-review.md). A good finding:

- names the file and line (`CURRICULUM.md:80`), or the notebook and cell;
- says **CONFIRMED** (reproduced: give the command, the executed cell or the quoted line) or **SUSPECTED**
  (plausible, not reproduced);
- carries a severity (H, M or L) when you file several;
- proposes the fix.

An unmarked number you cannot trace to code, a check that passes a wrong answer, and a T0 path that fails as
documented are all worth filing.

## Quick start for a first change

```bash
git clone https://github.com/aniryou/full-stack-agentic-engineer.git   # or your fork
cd full-stack-agentic-engineer
git switch -c my-first-change
python3 -m venv .venv && . .venv/bin/activate      # .venv/ is gitignored; use a fresh one per lab
cd 07-application-agent-framework/agent-fundamentals/agent-core
python3 -m pip install -e '.[dev]'                 # each lab's README gives its install line
python3 -m pytest -q                               # offline, no GPU, no keys
make check                                         # rebuild the notebooks, run the tests, the solutions and the blanks
```

Make the change (for a notebook, in `notebooks_src/`, then `make notebooks`), run the checks for what you touched,
commit, push and open a draft pull request. `make lab` opens the notebooks in JupyterLab. To read or run a notebook
without installing anything, open it from the "Run in Colab" links at the end of its layer README (for this lab,
[`07-application-agent-framework/README.md`](07-application-agent-framework/README.md)): its first cell clones the
repository and installs the lab.
