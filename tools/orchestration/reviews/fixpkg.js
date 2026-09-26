export const meta = {
  name: 'review-fix-package',
  description: 'Fix one package of adversarial-review findings on its own branch, verify adversarially, re-fix up to twice, open a draft PR',
  phases: [
    { title: 'Fix', detail: 'fixer works in its own worktree and venv, pushes a branch, opens a draft PR' },
    { title: 'Verify', detail: 'independent verifier reproduces each finding on main and confirms the fix on the branch' },
    { title: 'Re-fix', detail: 'fixer addresses the verifier defects' },
    { title: 'Re-verify', detail: 'verifier re-checks' },
  ],
}

const pkg = args.pkg
const common = args.common
const MODEL = common.model || 'opus'
const EFFORT = common.effort || 'high'

const FIX_SCHEMA = {
  type: 'object',
  properties: {
    branch: { type: 'string' },
    pr_url: { type: 'string' },
    pushed: { type: 'boolean' },
    findings: { type: 'array', items: { type: 'object', properties: {
      id: { type: 'string' },
      status: { type: 'string', enum: ['fixed', 'not-reproduced', 'deferred', 'partial'] },
      detail: { type: 'string' },
      commit: { type: 'string' },
    }, required: ['id', 'status', 'detail'] } },
    verification: { type: 'array', items: { type: 'object', properties: {
      command: { type: 'string' }, result: { type: 'string' },
    }, required: ['command', 'result'] } },
    notes_for_orchestrator: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
  required: ['branch', 'pushed', 'findings', 'verification', 'notes_for_orchestrator', 'summary'],
}

const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    pass: { type: 'boolean' },
    defects: { type: 'array', items: { type: 'object', properties: {
      finding_id: { type: 'string' }, file: { type: 'string' }, what: { type: 'string' }, reproduce: { type: 'string' },
    }, required: ['what', 'reproduce'] } },
    checks: { type: 'array', items: { type: 'object', properties: {
      check: { type: 'string' }, result: { type: 'string' },
    }, required: ['check', 'result'] } },
    notes: { type: 'array', items: { type: 'string' } },
  },
  required: ['pass', 'defects', 'checks', 'notes'],
}

const wt = `${common.scratch}/wt-${pkg.id}`
const venv = `${common.scratch}/venv-${pkg.id}`

function conventions() {
  return [
    'Repo conventions that apply to every edit:',
    '- Writing rule: content addresses an engineer explaining a design in a design review — never a role, an employer, a customer, a partner or an interviewer. No sales, MOU, partner-name, JD, rubric or interview wording.',
    '- Product facts and prices get a `(verify)` tag and a date; simulated numbers are labelled simulated, fixtures "sample output (illustrative)".',
    '- Run tiers: T0 laptop/Colab CPU/CI ($0, no GPU, no torch, no Docker, no keys) — every concept must be learnable at T0; T1 one small GPU; T2 multi-GPU box; T3 GCP via Terraform. GCP is one target, never a prerequisite.',
    '- Numbers in primers are pinned by tests (test_primer_numbers.py, test_docs.py and siblings): change text and test together.',
    '- READMEs follow tools/orchestration/README-STYLE.md (promise → start here → what you get with time and tier → run it → how it fits → caveats). Only touch READMEs inside your scope.',
    '- The environment here is Python 3.11, 4 shared vCPUs, no GPU, no Docker; PyPI is reachable. Do not install torch unless your findings say so.',
    '- Do NOT edit these (other packages own them; record needed changes in notes_for_orchestrator instead): ' + common.deny.join(', ') + (pkg.deny ? ', ' + pkg.deny.join(', ') : '') + '. Exceptions explicitly allowed for this package: ' + (pkg.allow_exceptions || 'none') + '.',
    '- Never put a model name in commit messages, PR text, code or comments.',
    '- Every commit message ends with these two lines exactly:',
    common.commit_trailer,
    '- The PR body ends with these lines exactly:',
    common.pr_trailer,
  ].join('\n')
}

function fixPrompt(prev) {
  const lines = []
  lines.push(`You are the fixer for work package ${pkg.id} — "${pkg.title}" — in the public learning repo aniryou/full-stack-agentic-engineer, checked out at ${common.repo} (do not modify that checkout: work only in your worktree).`)
  lines.push('')
  lines.push('SETUP (do exactly this):')
  lines.push(`1. Worktree on your own branch: if ${wt} does not exist: \`cd ${common.repo} && git fetch origin main && git worktree add ${wt} -b ${pkg.branch} origin/main\`. If it exists (a re-fix round): \`cd ${wt} && git fetch origin ${pkg.branch} && git status\` — it must be clean and on ${pkg.branch}; otherwise \`git checkout ${pkg.branch}\`.`)
  lines.push(`2. Your own venv: \`python3 -m venv ${venv} && . ${venv}/bin/activate && pip install -q -U pip wheel\` (reuse if it exists). Install each lab you work on the way its README says (pip install -e '<lab>[dev]' or -r requirements.txt). Do not create other venvs; disk and CPU are shared with other workers.`)
  lines.push(`3. Work from ${wt}. Every path below is relative to the repo root.`)
  lines.push('')
  lines.push(`SCOPE — files you own: ${pkg.scope}`)
  lines.push('')
  lines.push('FINDINGS TO FIX (from the adversarial review dated 2026-09-26; file:line references are against commit 3be2bb0 and may have shifted slightly):')
  lines.push(pkg.findings)
  lines.push('')
  if (pkg.notes) { lines.push('PACKAGE NOTES: ' + pkg.notes); lines.push('') }
  if (prev) {
    lines.push('THIS IS A RE-FIX ROUND. An independent verifier rejected the previous attempt. Its defects (each must be addressed or refuted with evidence):')
    lines.push(JSON.stringify(prev.defects, null, 1))
    lines.push('Verifier notes: ' + JSON.stringify(prev.notes))
    lines.push('')
  }
  lines.push('METHOD, per finding: (1) reproduce it first on your branch (run the command, execute the cell, read the line) — if it does not reproduce on current main, mark it not-reproduced with the evidence and move on; (2) make the smallest correct fix that keeps the repo conventions; (3) update pinned tests together with text; (4) add a regression test for every code bug; (5) never weaken or skip a test to get green.')
  lines.push('')
  lines.push('VERIFICATION you must run before pushing (all must pass; paste the tail of each result in your report):')
  lines.push(pkg.verify)
  lines.push(`Plus: \`python3 tools/orchestration/mdlinks.py <every .md you touched>\` must be clean, and \`git diff --name-only origin/main...HEAD\` must stay inside your scope.`)
  lines.push('')
  lines.push('COMMIT AND PUSH: small logical commits with clear messages. Push after the FIRST commit and after every further commit (`git push -u origin ' + pkg.branch + '`): a usage cap can end this session at any moment and only pushed work survives. Do not rebase or force-push.')
  lines.push('')
  lines.push(`PULL REQUEST: when everything is pushed, open a DRAFT pull request from ${pkg.branch} to main with the GitHub MCP tool (load it with ToolSearch "select:mcp__github__create_pull_request,mcp__github__list_pull_requests"; owner aniryou, repo full-stack-agentic-engineer). First list open PRs for the head branch; if one exists (re-fix round) do not create another. Title: "${pkg.title}". Body: what changed per finding (id → fix), how it was verified, what was not reproduced or deferred and why; then the trailer lines.`)
  lines.push('')
  lines.push(conventions())
  lines.push('')
  lines.push('REPORT (StructuredOutput): branch, pr_url, pushed, one entry per finding id with status fixed | not-reproduced | deferred | partial and a one-line detail (+ commit sha), the verification commands with results, notes_for_orchestrator (every change you needed in files outside your scope, precisely: file, line, new text; plus anything the orchestrator must know), and a 3-line summary. Keep it terse — it is data, not prose.')
  return lines.join('\n')
}

function verifyPrompt(fix) {
  const vwt = `${common.scratch}/wt-${pkg.id}-v`
  const mwt = `${common.scratch}/wt-${pkg.id}-main`
  const vvenv = `${common.scratch}/venv-${pkg.id}-v`
  const lines = []
  lines.push(`You are an independent, adversarial verifier for work package ${pkg.id} — "${pkg.title}" — in the repo aniryou/full-stack-agentic-engineer checked out at ${common.repo} (do not modify that checkout). The fixer claims the following (treat as claims, not facts):`)
  lines.push(JSON.stringify(fix, null, 1))
  lines.push('')
  lines.push('SETUP:')
  lines.push(`- Branch worktree (detached, read-only for you): \`cd ${common.repo} && git fetch origin ${pkg.branch} main && (test -d ${vwt} && (cd ${vwt} && git checkout --detach origin/${pkg.branch}) || git worktree add --detach ${vwt} origin/${pkg.branch})\`.`)
  lines.push(`- Baseline worktree: \`test -d ${mwt} || git worktree add --detach ${mwt} origin/main\` — use it to reproduce each original defect before checking the fix.`)
  lines.push(`- Fresh venv: \`python3 -m venv ${vvenv} && . ${vvenv}/bin/activate && pip install -q -U pip wheel\` (reuse if it exists). Install labs exactly as their READMEs instruct (that is part of what you verify).`)
  lines.push('')
  lines.push('THE FINDINGS the fixer had to address:')
  lines.push(pkg.findings)
  lines.push('')
  lines.push('WHAT TO CHECK (be strict; pass=false if anything fails):')
  lines.push('1. For every finding marked fixed: reproduce the original defect on the baseline worktree (run it, do not just read), then run the same probe on the branch worktree and confirm it is gone. For check-cell or assert changes, try at least one wrong answer and confirm it now fails. For text fixes, confirm the new wording is correct (recompute numbers with the cores; check upstream sources when reachable) and follows the writing rule (design-review audience; no customer/partner/interview/sales wording; `(verify)` on product facts).')
  lines.push('2. For every finding marked not-reproduced or deferred: check the evidence; a finding that does reproduce on main is a defect.')
  lines.push('3. Run every verification command below in your own venv and record the tail of each result:')
  lines.push(pkg.verify)
  lines.push('4. `python3 tools/orchestration/mdlinks.py` on every .md changed by the branch must be clean; `git diff --name-only origin/main...origin/' + pkg.branch + '` must stay inside the package scope: ' + pkg.scope + ' — anything outside it is a defect unless listed as an allowed exception: ' + (pkg.allow_exceptions || 'none') + '.')
  lines.push('5. No test was weakened, skipped or deleted; every code fix has a regression test that fails on the baseline (check by running the new test against the baseline code where practical).')
  lines.push('6. Commit messages carry the two trailer lines; no model names anywhere in the diff; the PR exists as a draft against main if the fixer says so (load `mcp__github__list_pull_requests` via ToolSearch to check).')
  lines.push('7. Look for regressions the fixer may have introduced: read the full diff (`git diff origin/main...origin/' + pkg.branch + '`) adversarially.')
  lines.push('')
  lines.push('Do NOT fix anything and do NOT push. Return pass (true only if every check passed), defects (finding_id, file, what is wrong, exact command to reproduce), checks (each check and its result) and notes (residual risks, out-of-scope changes the orchestrator must make). Terse; it is data.')
  return lines.join('\n')
}

phase('Fix')
let fix = await agent(fixPrompt(null), { label: `fix:${pkg.id}`, phase: 'Fix', schema: FIX_SCHEMA, model: MODEL, effort: EFFORT })
if (!fix) return { id: pkg.id, branch: pkg.branch, status: 'fixer-died', pass: false }
log(`${pkg.id}: fixer done — pushed=${fix.pushed} pr=${fix.pr_url || 'none'}`)

let verify = await agent(verifyPrompt(fix), { label: `verify:${pkg.id}`, phase: 'Verify', schema: VERIFY_SCHEMA, model: MODEL, effort: EFFORT })
let rounds = 1
while (verify && !verify.pass && rounds < 3) {
  log(`${pkg.id}: verifier found ${verify.defects.length} defect(s) — re-fix round ${rounds}`)
  const refix = await agent(fixPrompt(verify), { label: `refix${rounds}:${pkg.id}`, phase: 'Re-fix', schema: FIX_SCHEMA, model: MODEL, effort: EFFORT })
  if (!refix) break
  fix = refix
  verify = await agent(verifyPrompt(fix), { label: `reverify${rounds}:${pkg.id}`, phase: 'Re-verify', schema: VERIFY_SCHEMA, model: MODEL, effort: EFFORT })
  rounds++
}

return {
  id: pkg.id,
  branch: pkg.branch,
  pr_url: fix.pr_url || null,
  pushed: fix.pushed,
  pass: verify ? verify.pass : null,
  rounds,
  findings: fix.findings.map(f => `${f.id}: ${f.status} — ${f.detail}`),
  notes_for_orchestrator: fix.notes_for_orchestrator,
  verifier_defects: verify ? verify.defects : ['verifier died'],
  verifier_notes: verify ? verify.notes : [],
  summary: fix.summary,
}
