// Review workflow for one layer's deliverables (primer + core + lab, or a standalone primer).
// Run with the Claude Code Workflow tool: args = { layer, title, topicDir, dirs: [...], specBlock, notes }.
// Shape: 3 adversarial reviewers with distinct lenses (barrier: the fixer needs every finding) → a fixer that
// must VERIFY each finding before acting → an independent validator that re-runs everything and audits the
// fixer's rejections. All agents run on Opus.
export const meta = {
  name: 'review-layer',
  description: 'Adversarial review of one layer: 3 lens reviewers → verifying fixer → independent validator',
  phases: [
    { title: 'Review', detail: 'concepts · runnability · pedagogy' },
    { title: 'Fix', detail: 'verify each finding, fix, re-validate' },
    { title: 'Validate', detail: 'independent re-run + audit of rejections' },
  ],
}

const SP = '/tmp/claude-0/-home-user-full-stack-agentic-engineer/4b9a43de-09d5-5f71-ab43-1b2252b82aa6/scratchpad'
const REPO = '/home/user/full-stack-agentic-engineer'
const L = args
const dirs = L.dirs.join(', ')

const FINDING = {
  type: 'object',
  properties: {
    id: { type: 'string' },
    severity: { type: 'string', enum: ['blocking', 'major', 'minor'] },
    category: { type: 'string' },
    file: { type: 'string' },
    location: { type: 'string' },
    claim: { type: 'string', description: 'what is wrong, precisely' },
    evidence: { type: 'string', description: 'how you know: the recomputation, the command output, the source you checked' },
    fix: { type: 'string', description: 'the concrete change that would resolve it' },
  },
  required: ['id', 'severity', 'category', 'file', 'claim', 'evidence', 'fix'],
}
const FINDINGS = {
  type: 'object',
  properties: { findings: { type: 'array', items: FINDING }, summary: { type: 'string' } },
  required: ['findings', 'summary'],
}
const FIX_REPORT = {
  type: 'object',
  properties: {
    fixed: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, what: { type: 'string' } }, required: ['id', 'what'] } },
    rejected: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, reason: { type: 'string' } }, required: ['id', 'reason'] } },
    deferred: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, reason: { type: 'string' } }, required: ['id', 'reason'] } },
    validation: { type: 'string', description: 'one line per validation command with its result' },
  },
  required: ['fixed', 'rejected', 'deferred', 'validation'],
}
const VALIDATION = {
  type: 'object',
  properties: {
    status: { type: 'string', enum: ['pass', 'fail'] },
    validation: { type: 'string' },
    unjustified_rejections: { type: 'array', items: { type: 'string' } },
    extra_fixes: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
  required: ['status', 'validation', 'unjustified_rejections', 'extra_fixes', 'notes'],
}

const COMMON = `Repo: ${REPO}. Target: layer ${L.layer} — ${L.title}. Topic dir: ${L.topicDir}. Deliverable dirs: ${dirs}.
Contract: ${SP}/SPEC.md (§0–5 conventions; §6 block "${L.specBlock}" is what this layer promised; §4 validation). Facts: ${SP}/FACTS.md.
Extra notes from the orchestrator: ${L.notes || 'none'}.
Severity: blocking = technically wrong or misleading content, a failing/broken artifact, a SPEC-required deliverable missing, simulated or fixture numbers presented as measurements, a Colab bootstrap that cannot work; major = a SPEC-required section or notebook that is shallow or incomplete, tautological or missing tests for a core claim, an exercise that does not test understanding, wrong tier labelling, an unverified product fact not marked (verify), inconsistency between primer/core/lab; minor = clarity, style, naming.
Be adversarial and specific: every finding needs evidence (your recomputation, the command output, the upstream source you checked — raw.githubusercontent.com works, most doc sites are blocked). Do not report style preferences as findings. Do NOT edit any file. Return only the structured findings.`

phase('Review')
log(`Layer ${L.layer}: three reviewers on ${L.dirs.length} dirs`)
const reviews = await parallel([
  () => agent(`${COMMON}
Your lens: TECHNICAL CORRECTNESS. Read the primer and every library module. Recompute every formula and worked number by hand (state your arithmetic); check units and orders of magnitude; check hardware/software claims against FACTS.md and, where they matter, against upstream source; check that the code implements what the primer says (and vice versa); check simulator/model semantics against how the real system behaves (Kubernetes, Kueue, vLLM, NCCL, CUDA, HPA, Gateway API — whichever apply); check that anything not measured is labelled simulated/assumed; check dates and (verify) marks on product facts. Read tests: do they pin real values or restate the code?`,
    { label: `review:${L.layer}:concepts`, phase: 'Review', schema: FINDINGS, model: 'opus', effort: 'high' }),
  () => agent(`${COMMON}
Your lens: RUNNABILITY AND ENGINEERING. Actually run the SPEC §4 validation for every deliverable dir (pytest; tools/build_notebooks.py; tools/run_notebooks.py solutions; tools/run_notebooks.py notebooks --expect-fail; ${SP}/tfcheck.sh on any Terraform dir; kubernetes-validate --strict -k 1.34.0 on core-kind YAML; bash -n on shell scripts; python3 ${SP}/mdlinks.py on the topic dir) and report exact failures. Check the notebook bootstrap cell: the repo-relative path it cd's into must be the deliverable's real location, and the package must be importable after \`pip install -e .\` (test in a temp venv: python3 -m venv /tmp/venv-${L.layer} && /tmp/venv-${L.layer}/bin/pip install -q -e <dir>[dev] then run pytest there — dependencies missing from pyproject are blocking). Check pyproject/requirements/Makefile/README commands agree; junk files (__pycache__, .pytest_cache, *.egg-info, _run_outputs, .terraform*, .ipynb_checkpoints); absolute /tmp or scratchpad paths leaked into repo files; tests that would pass with the implementation deleted; GPU/torch/Docker/cloud paths guarded so the T0 path runs anywhere; scripts have set -euo pipefail and DRY_RUN; .gitignore present; LICENSE present; notebook structure per SPEC §3 (tier line, one-minute version, exercises with checks, design-review close). Clean up any venv you created.`,
    { label: `review:${L.layer}:runnability`, phase: 'Review', schema: FINDINGS, model: 'opus', effort: 'high' }),
  () => agent(`${COMMON}
Your lens: DEPTH, PEDAGOGY AND CONSISTENCY. The learner wants deep understanding. Compare what was built against every item in the SPEC §6 block: is each promised section/module/notebook present and treated with real depth (mechanism, worked numbers, failure modes, trade-offs), or padded and hand-wavy? Do exercises make the learner implement the key idea or predict a number, rather than fill boilerplate? Are checks meaningful? Does the "In a design review" close give a coherent 2-minute walkthrough and drills with correct answers? Are the primer's section numbers/titles the ones the lab README and notebooks cite? Do primer, core README and lab README agree on names, numbers and claims? Is existing repo material linked instead of duplicated (SPEC §5 list)? Is GCP presented as optional with real non-GCP paths, and is the tier of each notebook right? Flag missing prerequisites and places where a strong engineer would be misled or left with a wrong mental model.`,
    { label: `review:${L.layer}:pedagogy`, phase: 'Review', schema: FINDINGS, model: 'opus', effort: 'high' }),
])
const all = reviews.filter(Boolean).flatMap((r, i) => r.findings.map(f => ({ ...f, id: `${['C', 'R', 'P'][i]}-${f.id}` })))
const counts = { blocking: 0, major: 0, minor: 0 }
for (const f of all) counts[f.severity] = (counts[f.severity] || 0) + 1
log(`Layer ${L.layer}: ${all.length} findings (blocking ${counts.blocking}, major ${counts.major}, minor ${counts.minor})`)

phase('Fix')
const fix = await agent(`${COMMON.replace('Do NOT edit any file. Return only the structured findings.', '')}
You are the FIXER. Below are ${all.length} findings from three reviewers (ids prefixed C=concepts, R=runnability, P=pedagogy). For EACH finding: first verify it yourself (reproduce the failure, redo the arithmetic, read the code/source); if it is real, fix it properly (root cause, not a patch that hides it; never delete or weaken a test to pass; keep primer↔code↔tests consistent — if you change a number in code, update the primer and its pinning test); if it is wrong, reject it with a precise reason; defer only what genuinely needs hardware/cloud you do not have, saying so. Fix all blocking and major findings; fix minor ones when cheap. You may edit only files under: ${dirs} and ${L.topicDir}/PRIMER.md, ${L.topicDir}/README.md. Do not touch layer READMEs, root files or other layers. Then re-run the full SPEC §4 validation for every deliverable dir and clean build junk (__pycache__, .pytest_cache, *.egg-info, _run_outputs, .terraform*, .ipynb_checkpoints, any venv). Report with the schema; \`validation\` must quote one result line per command.

FINDINGS (JSON):
${JSON.stringify(all, null, 1)}`,
  { label: `fix:${L.layer}`, phase: 'Fix', schema: FIX_REPORT, model: 'opus', effort: 'xhigh' })
log(`Layer ${L.layer}: fixed ${fix ? fix.fixed.length : 0}, rejected ${fix ? fix.rejected.length : 0}, deferred ${fix ? fix.deferred.length : 0}`)

phase('Validate')
const val = await agent(`${COMMON.replace('Do NOT edit any file. Return only the structured findings.', '')}
You are the INDEPENDENT VALIDATOR, after a fixer edited the tree. (1) Re-run the full SPEC §4 validation yourself for every deliverable dir (pytest; build + run notebooks both modes; tfcheck; kubernetes-validate; bash -n; mdlinks) and report each result line. (2) Audit the fixer's REJECTED findings below: for each, decide whether the rejection reason holds (inspect the code/primer; redo arithmetic); list the ids whose rejection is NOT justified and fix those yourself. (3) Skim \`git diff -- ${dirs} ${L.topicDir}\` for damage the fixes may have caused (deleted tests, weakened asserts, numbers changed in code but not in the primer, broken links). (4) Fix small breakages yourself; anything larger goes in notes. Clean build junk. Report with the schema: status is 'pass' only if every validation command passes and no blocking finding remains open.

FIXER REPORT (JSON):
${JSON.stringify(fix, null, 1)}`,
  { label: `validate:${L.layer}`, phase: 'Validate', schema: VALIDATION, model: 'opus', effort: 'high' })

return {
  layer: L.layer,
  findings: all.length,
  counts,
  fixed: fix ? fix.fixed.length : null,
  rejected: fix ? fix.rejected : null,
  deferred: fix ? fix.deferred : null,
  validation: val,
}
