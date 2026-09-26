// Build one topic end to end: research facts → builder A (PRIMER + README + core) ∥ builder B (lab) → nested adversarial review.
// args = { topic, layer, title, topicDir, coreDir, corePkg, labDir, labPkg, specBlock, existing, researchBrief, sp, repo, notes }
export const meta = {
  name: 'build-topic',
  description: 'One topic: research facts → build primer+core (A) and lab (B) in parallel → adversarial review, fix, validate',
  phases: [
    { title: 'Research', detail: 'verify product facts from cloned upstream sources', model: 'opus' },
    { title: 'Build', detail: 'A: PRIMER + README + core · B: lab', model: 'opus' },
    { title: 'Review', detail: 'nested review workflow', model: 'opus' },
  ],
}

const T = args
const SP = T.sp
const REPO = T.repo
const FACTS_FILE = `${SP}/facts-${T.topic}.md`

const RESEARCH_REPORT = {
  type: 'object',
  properties: {
    file: { type: 'string' },
    facts_count: { type: 'integer' },
    unverified: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
  required: ['file', 'facts_count', 'unverified', 'notes'],
}
const BUILD_REPORT = {
  type: 'object',
  properties: {
    status: { type: 'string', enum: ['done', 'partial'] },
    files: { type: 'string', description: 'file count and the directories created' },
    tests: { type: 'string', description: 'the pytest one-liner result' },
    notebooks: { type: 'string', description: 'solutions x/y passed; blanks x/y stopped at the first exercise' },
    tf: { type: 'string', description: 'tfcheck result or n/a' },
    k8s: { type: 'string', description: 'kubernetes-validate result or n/a' },
    links: { type: 'string', description: 'mdlinks result' },
    deviations: { type: 'string', description: 'anything that differs from SPEC and why' },
    open: { type: 'string', description: 'VERIFY items, things not validated here (GPU/Docker/cloud), follow-ups' },
  },
  required: ['status', 'files', 'tests', 'notebooks', 'tf', 'k8s', 'links', 'deviations', 'open'],
}

const ENV = `Environment and rules (read ${SP}/FACTS.md "Environment" section for the full list): Python 3.11; numpy, pytest, nbformat, nbclient, ipykernel, matplotlib, pyyaml, aiohttp, httpx, kubernetes-validate and torch 2.14 (CPU) are installed system-wide; no GPU, no Docker daemon, no kind/kubectl/helm; 4 CPUs and 15 GB RAM shared with other agents (keep tests < 60 s, notebook runs short). Installs only via \`${SP}/pipi <pkgs>\` (serialized) — never torch/vllm/triton/flash-attn, never anything that downloads model weights. Terraform: \`export ORCH_SCRATCH=${SP} TF_CLI_CONFIG_FILE=${SP}/tf/terraformrc TERRAFORM_BIN=${SP}/tf/bin/terraform; ${SP}/tfcheck.sh <dir>\`; attribute lookups \`TF_SCHEMA_JSON=${SP}/tf/schema.json python3 ${SP}/tfattrs.py <resource> [filter]\`. Upstream sources are cloned under ${SP}/ref/ (grep them; doc web sites, arxiv, huggingface.co are blocked; raw.githubusercontent.com and git clone work). NEVER run git add/commit/checkout/stash/reset — the orchestrator commits. Never write outside your assigned directories. No emojis except ✅ in check output. Writing rule: address an engineer explaining a design in a design review — never a role or an employer.`

const COMMON = `Repo: ${REPO} (its CLAUDE.md is already in your context). Topic: ${T.title} — layer ${T.layer}, topic dir ${T.topicDir}.
Contract: ${SP}/SPEC.md — §0–5 (goal, tiers, layout, conventions, validation, primer contract), §6b's opening list of differences for this build, and the §6b block "${T.specBlock}" (what this topic must deliver; keep its section numbers and titles, module names, notebook names and deploy targets). Facts: ${SP}/FACTS.md and the topic fact sheet ${FACTS_FILE}. README style: ${SP}/README-STYLE.md. Existing repo material this topic must cite and reuse rather than duplicate: ${T.existing}.
${ENV}`

phase('Research')
log(`${T.topic}: researching facts → ${FACTS_FILE}`)
const research = await agent(`You are the RESEARCHER for the topic "${T.title}". Produce the dated fact sheet ${FACTS_FILE} that the two builders and three reviewers will treat as the source of truth (≤ 450 lines, Markdown, sections with short bullet facts, each with its source path).
Read first: ${SP}/SPEC.md §6b block "${T.specBlock}" (so you know exactly which facts the primer, core and lab need), ${SP}/FACTS.md (what is already verified; do not repeat it, extend it), and the existing repo files named in "Must cite/reuse" so you can record their exact function names, section numbers and numbers.
Then verify, from the sources cloned under ${SP}/ref/ (grep -rn; read the actual code and docs; cite file paths) or raw.githubusercontent.com, every product and technical fact in this brief — flags and their defaults, API fields, config values, model architecture numbers computed from configs, formulas as implemented in code (quote the code), version numbers, minimum hardware requirements, and the pitfalls a builder would otherwise get wrong:
${T.researchBrief}
Rules: quote exact identifiers (flag names, enum values, class names, config keys) — do not paraphrase them; compute derived numbers (parameter counts, bytes) and show the arithmetic; anything you could not find in a source is written as (unverified) with your best knowledge and why; note where the repo's existing material already states a number so builders stay consistent; end with a "Model and tool ids to use" list (T0/T1 choices with sizes) and a "Pitfalls" list. Do not write anything outside ${SP}/. Return the structured summary.`,
  { label: `research:${T.topic}`, phase: 'Research', schema: RESEARCH_REPORT, model: 'opus', effort: 'high' })
log(`${T.topic}: facts ${research ? research.facts_count : 'n/a'}, unverified ${research ? research.unverified.length : 'n/a'}`)

phase('Build')
log(`${T.topic}: builder A (primer + README + ${T.coreDir}) ∥ builder B (${T.labDir})`)
const [a, b] = await parallel([
  () => agent(`${COMMON}
You are BUILDER A. Deliver, exactly per the §6b block: ${T.topicDir}/README.md (topic index in README-STYLE shape: promise → start here → what you get with time and tier → run it → how it fits → caveats), ${T.topicDir}/PRIMER.md (SPEC §5 contract: 600–900 lines, the numbered sections with the block's titles, worked numbers each naming the core function that computes it, "In a design review" with a 2-minute walkthrough and 6 drills with answers, Glossary, Sources, dated Verify list), and the core ${T.coreDir}/ (package ${T.corePkg}: standard library + numpy, offline, 500–1,000 lines of readable library code, each module opening with the one idea it teaches; the block's modules and notebooks; tests that pin formulas to hand-computed values and reproduce the existing repo numbers the block names).
Before writing: read ${SP}/SPEC.md fully, ${SP}/FACTS.md, ${FACTS_FILE}, ${SP}/README-STYLE.md; copy the packaging and notebook tooling from ${REPO}/07-application-agent-framework/agent-fundamentals/agent-core (pyproject, Makefile, LICENSE, .gitignore, tools/build_notebooks.py, tools/run_notebooks.py — change only BOOTSTRAP's repo-relative dir and package name) and study one finished core of the same shape for depth and voice: ${REPO}/04-inference-engine/serving-engine/mini-engine-core (its README, notebooks_src/01 and 06, tests/test_quant.py) and ${REPO}/01-hardware-gpu-fabric/roofline-and-fabric/roofline-core (notebooks_src/02, tests/test_primer_numbers.py). Match that depth: mechanisms, worked numbers, failure modes, exercises that make the learner implement the idea or predict a number, checks that print ✅.
The lab is being built in parallel by builder B from the same block; it will cite your primer sections by the block's numbers and titles, so keep them exactly. Do not write anything under ${T.labDir}.
Then run the SPEC §4 validation for the core (pytest; build_notebooks; run_notebooks solutions; run_notebooks notebooks --expect-fail; mdlinks on ${T.topicDir}), fix until clean, and remove build junk (__pycache__, .pytest_cache, *.egg-info, _run_outputs, .ipynb_checkpoints). Report with the schema (validation lines quoted verbatim).`,
    { label: `build:${T.topic}:A`, phase: 'Build', schema: BUILD_REPORT, model: 'opus', effort: 'xhigh' }),
  () => agent(`${COMMON}
You are BUILDER B. Deliver the lab ${T.labDir}/ (package ${T.labPkg}) exactly per the §6b block: the modules, notebooks and deploy targets it lists, with T0 fallbacks that run offline on CPU (torch paths lazy and skipped when absent; Docker/kind/cloud paths detect absence, print the exact commands and use bundled sample output labelled "sample output in the documented format (illustrative)"; simulated timings labelled simulated), the T1/T2/T3 paths written carefully even though you cannot run them here, a README in README-STYLE shape with a tier table, tests (offline, no GPU, < 60 s) including manifest/Terraform/config checks, and deploy READMEs with cost and cleanup.
Before writing: read ${SP}/SPEC.md fully, ${SP}/FACTS.md, ${FACTS_FILE}, ${SP}/README-STYLE.md; copy the packaging and notebook tooling from ${REPO}/07-application-agent-framework/agent-fundamentals/agent-core (pyproject, Makefile, LICENSE, .gitignore, tools/build_notebooks.py, tools/run_notebooks.py — change only BOOTSTRAP's repo-relative dir and package name) and study one finished lab of the same shape for depth, structure and voice: ${REPO}/04-inference-engine/serving-engine/vllm-serving-lab (README, servelab/fakeserver.py, notebooks_src/02 and 05, tests/test_deploy.py, deploy/) and, for Kubernetes or kind material, ${REPO}/03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab (k8sgpu/manifests.py, deploy/kind/, tests). Match that depth.
The primer and core are being written in parallel by builder A from the same block: cite primer sections by the block's numbers and titles (e.g. "PRIMER §4 ..."), never import the core package (labs are standalone; small helpers may be re-implemented), and do not write anything under ${T.coreDir} or ${T.topicDir}/PRIMER.md or ${T.topicDir}/README.md.
Then run the SPEC §4 validation for the lab (pytest; build_notebooks; run_notebooks solutions; run_notebooks notebooks --expect-fail; tfcheck on any Terraform dir; kubernetes-validate --strict -k 1.34.0 on core-kind YAML; bash -n on scripts; mdlinks on ${T.labDir}), fix until clean, and remove build junk (__pycache__, .pytest_cache, *.egg-info, _run_outputs, .terraform*, .ipynb_checkpoints). Report with the schema (validation lines quoted verbatim).`,
    { label: `build:${T.topic}:B`, phase: 'Build', schema: BUILD_REPORT, model: 'opus', effort: 'xhigh' }),
])
log(`${T.topic}: A ${a ? a.status : 'null'} · B ${b ? b.status : 'null'}`)

phase('Review')
const review = await workflow({ scriptPath: `${SP}/review_workflow.js` }, {
  topic: T.topic, layer: T.layer, title: T.title, topicDir: T.topicDir, dirs: [T.coreDir, T.labDir],
  specBlock: T.specBlock, sp: SP, repo: REPO,
  notes: `Builder A report: ${JSON.stringify(a)}. Builder B report: ${JSON.stringify(b)}. ${T.notes || ''}`,
})

return { topic: T.topic, research, builderA: a, builderB: b, review }
