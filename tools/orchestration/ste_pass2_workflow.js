// The STE-100 second pass, one layer per run: `args = {layer, items}` with the same disjoint file groups as the first
// pass (kind: md, layer-readme, pct, nb, builder), each with the counts of the linter's second-pass findings. The
// prose is already in STE; this pass applies the dictionary-level word list and the paren-clause rule of
// STE100-STYLE.md 3.1 (second table) and reads for plain language. One agent applies, an adversarial verifier reads the
// diff against git HEAD hunk by hunk, a fixer repairs, and a re-check follows, up to two fix rounds.
export const meta = {
  name: 'ste100-pass2-layer',
  description: 'STE second pass over one layer: dictionary-level words, parenthetical clauses, plain language; verify the diff; fix',
  phases: [
    { title: 'Apply', detail: 'one agent per file group: the linter findings and a plain-language read', model: 'opus' },
    { title: 'Verify', detail: 'the diff against git ${BASE}, hunk by hunk, plus the checks', model: 'opus' },
    { title: 'Fix', detail: 'repair the findings, re-run the checks', model: 'opus' },
  ],
}

const R = '/home/user/full-stack-agentic-engineer'
const S = '/tmp/claude-0/-home-user-full-stack-agentic-engineer/54928949-5727-58e1-ab35-871db6a3c519/scratchpad'
const STYLE = R + '/tools/orchestration/STE100-STYLE.md'
const LINT = 'python3 ' + R + '/tools/orchestration/ste_lint.py'
const MDLINKS = 'python3 ' + R + '/tools/orchestration/mdlinks.py'
const NBMD = 'python3 ' + R + '/tools/orchestration/nb_md.py'
const NBOUT = 'python3 ' + R + '/tools/orchestration/nb_outputs.py'
const NBSRC = 'python3 ' + R + '/tools/ci/nb_sources.py --check'
const INJECT = 'python3 ' + R + '/tools/inject_colab_bootstrap.py'

const LAYER = args.layer
const ITEMS = args.items
// The text the agents compare against: git HEAD by default, or a commit that already holds other passes' edits to the
// same files (args.baseline), so that those edits are not read as this pass's.
const BASE = args.baseline || 'HEAD'

const doneResolve = {}
const donePromise = Object.fromEntries(ITEMS.map(i => [i.key, new Promise(res => { doneResolve[i.key] = res })]))
const partNote = (item) => item.sections
  ? `\n\nThis item covers only part of its file: ${item.sections}. Edit the prose of those sections only; every line outside them stays byte-identical to what it is now (another item owns the other sections${item.after ? ' and edited them before you started' : ' and edits them after you finish'}). Use the headings as the boundaries. The linter and the checks run on the whole file; a finding outside your sections is not yours to repair.`
  : ''
const partVerifyNote = (item) => item.sections
  ? `\n\nThis item covers only part of its file: ${item.sections}. Review the hunks in those sections only. The other sections ${item.after ? 'were edited by an earlier item and are not under review' : 'are not under review but must be byte-identical to the original: any hunk there is blocking'}.`
  : ''

const COMMON = `Repository: ${R} (branch claude/asd-ste100-agent-rewrite-hfcxty). File paths below are relative to the repository root; the absolute path is ${R}/<path>. Use absolute paths in every command; when a command must run inside a lab, write "cd <absolute dir> && <command>" in that one command. The current text of every prose file is at git ${BASE}: read it with "git show ${BASE}:<relative path>", and see your changes with "git diff -U1 ${BASE} -- <absolute path>". Never run a git command that changes state (add, commit, stash, checkout, restore, reset, clean, rebase, merge, push): many agents work in this same checkout at the same time, on other files. Never pip install anything. Temporary files go under ${S}. Never put a model name or model identifier in any file. The style brief is ${STYLE}; read it completely before you start, in particular section 1 (what does not change, items 11 to 14 included), section 2 (the writing rules), section 3.1 (the two tables of replacements and the rule on parentheses), 3.2 to 3.4 (the technical names of this layer) and 3.5 (document types).`

const APPLY_RULES = `TASK: the prose of the files below is already in ASD-STE100 (a first pass rewrote it; the linter reports 0 errors except the fragments that tests pin and the quotations). Make it easier to read, in place, by applying the second-pass rules of the brief; the meaning stays exactly the same.

What to do, in this order:
1. Run "${LINT} <file>" on each file and read every finding. The findings that matter in this pass are:
   - word: a word the standard replaces (the two tables of brief 3.1). Replace it with the plain word that keeps the meaning. Keep it only when it is a technical name of this layer (brief 3.2 to 3.4: a display, a state, a fixed cost, an all-reduce, a major version, a verified reward, a tiny model named so in the code, the (verify) tag), a product or API name, a quotation, a fragment that a listed test pins, or the replacement would change the meaning ("since" as a time, "little" in "Little's law", "optimum" as the noun of an optimisation). Record each kept word in kept_warnings with the reason, once per word per file.
   - paren-clause: a clause with a verb inside parentheses in the middle of a sentence. Make it a sentence of its own after the sentence it comments on (in STE: at most 25 words, active, no modal), keep it when the parentheses hold a cross-reference, a value, a unit, a code span or a (verify) tag, or drop it when the sentence already says it. Never drop a fact.
   - semicolon, passive, ing: repair the ones that have a plain repair; keep a technical name (brief 3.2) or a state and say so in kept_warnings.
2. Then read each file once as a reader who meets the topic for the first time, and repair what the linter cannot see, with the smallest change: a cluster of more than three nouns (break it with "of", "for" or a verb); an unclear "it", "this", "that", "they" or "which" (name the thing); two ideas in one sentence (two sentences); a word used with two meanings in one paragraph (one word, one meaning); a long word where a short one says the same (the standard's approved words, brief 3.1 last paragraph); a sentence that names the result before its cause (cause first, then "thus"). Do not reflow paragraphs, do not merge or split paragraphs, do not move text.
3. Every rule of section 1 of the brief holds: headings, fenced blocks, inline code, formulas, link targets, tables' shapes, numbers, dates, (verify) tags, hedges, quotations and the pinned fragments stay verbatim; a bold label that leads a paragraph stays; a technical name is never replaced and never used as a verb; nothing is added or removed in meaning, no claim gets stronger or weaker.
4. After each file run "${LINT} <file>" again: the error count is not higher than before (compare with the original: "git show ${BASE}:<path> > ${S}/<name>.orig.md" then lint that copy); every warning you kept is in kept_warnings with its reason. Run every check listed for your files. A test that reports a missing fragment ("PRIMER.md no longer says: ...") means that fragment must go back verbatim; never change a test.
5. Read "git diff -U1 ${BASE} -- <each file>" once, hunk by hunk: each hunk is a word, a phrase or one sentence changed, with the same meaning.
6. Report with the structured output: files (the files you changed), lint (the counts of your last linter run, summed over your files), applied (counts: word, paren_clause, other), kept_warnings (word and reason), rule_breaks (a place where you kept the meaning and broke an STE rule, with the reason), unsure (anything you could not resolve), checks_run (each command and its result line).
Do not edit any file that is not in your list. Do not touch code cells, fenced code blocks, comments in code, Mermaid blocks, SVG files, figure captions' facts, the Colab cell or Python code outside Markdown cells.`

const VERIFY_RULES = `TASK: you are an adversarial verifier. Another agent applied the STE second pass to the prose of the files below, in place; the text before its edits is at git ${BASE}. Your job is to find every way the edits are wrong, hunk by hunk: run "git diff -U2 ${BASE} -- <absolute file>" for each file and read every hunk against its context in the original ("git show ${BASE}:<relative path>"). Assume something is wrong and look for it; do not praise.
Severity:
 blocking: a hunk that changes a fact, a number, a unit, a limit, a date, a (verify) tag, a hedge ("roughly", "about", "best-effort"), a cross-reference, a quotation or a claim (lost, added, stronger or weaker); a heading, link target, fenced block, inline code span, formula, table row or column, list item, Mermaid block, figure or caption fact that changed; a technical name of the layer (brief 3.2 to 3.4) replaced by a plain word, or a plain word that now reads as a different technical thing; a replacement that changed the meaning ("since 1.17.0" to "because 1.17.0", "within" to "in" where the meaning was "not more than", "reduce" to "decrease" where the subject is a collective such as all-reduce); a failing test or check listed for the files; a notebook that no longer builds; text edited inside a generated section, a code cell or a comment; a model name or identifier in a file; a paragraph reflowed, merged, split or moved.
 major: an STE rule broken by the new text: a sentence over 25 words, a modal, a contraction, a dash or arrow as a word, an -ing verb, a passive in an instruction, a cluster of more than three nouns, an unclear "it", "this", "that" or "which", a parenthetical comment that became a sentence but lost its link to what it comments on, two sentences where the cause and the result were separated so that the reasoning is gone; a kept warning whose reason is wrong (a plain word kept as if it were a technical name); a replacement that reads worse than the original to a first-time reader (say why).
 minor: a linter WARN that could still be repaired; an awkward sentence; a list item that no longer parallels its neighbours.
Method: for each file run "${LINT} <file>" and compare its error count with the original's (lint "git show ${BASE}:<path>" written to ${S}); run every check listed for the files; read the whole diff of every file, every hunk, not a sample; then sample ten sentences the agent did not change and say whether a second-pass finding was left there without a reason in kept_warnings. For each finding give the file, the location (the nearest heading and a quote of the new sentence), the original text, the problem and the exact replacement text. At most 60 findings per file, most severe first, no duplicates. Count the hunks you read. Do not edit any file.
Verdict: "fix-needed" when any blocking or major finding exists, else "pass".`

const FIX_RULES = `TASK: you are the fixer for the STE second pass of the files below. A verifier reported the findings in the JSON at the end. For each finding: read the current text and the original ("git show ${BASE}:<path>"); if the finding is real, repair it with the smallest change that keeps the text in STE; if it is not real, or the fix would break section 1 of the brief (facts, code, headings, link targets, structure, quotations, meaning, pinned fragments), reject it with a one-line reason. Also repair real problems the verifier missed when you see them. After the fixes run the linter on each file (no more errors than the original; every kept warning justified) and every check listed for your files, and re-read "git diff -U1 ${BASE} -- <file>" once more.
Report with the structured output: fixed (finding id and what you changed), rejected (finding id and why), lint (the counts of your last run), checks_run (each command and its result line).`

const kindNotes = (item) => {
  const lab = item.lab ? `${R}/${item.lab}` : null
  switch (item.kind) {
    case 'layer-readme':
      return `This is a layer README (brief, section 3.5). Keep the H1 verbatim. Never edit anything between the "<!-- colab-links -->" markers: that section belongs to tools/gen_colab_index.py. The stack diagram in a code block is code. Rewrite the prose outside the markers, the "What is inside" table cells as short sentences. Check: "${MDLINKS} <file>" gives 0 broken links, and "cd ${R} && python3 tools/gen_colab_index.py && git diff --stat" shows no change beyond your own prose edits in that file (the generator must stay a no-op).`
    case 'md':
      return `Markdown documents (brief, section 3.5: a primer, a README, docs). Headings, code blocks, link targets, tables' shapes, numbers and (verify) tags verbatim. Check each file with "${MDLINKS} <file>" (0 broken links).`
    case 'pct':
      return `Percent-format notebook sources (brief, section 1 item 12). Edit only the "# %% [markdown]" cells: one "# " comment per line, keep that prefix on every line, keep every heading inside its Markdown cell, keep the first cell's H1 and the "**Tier:**" line's facts. The "# %%", "# %% exercise" and "# %% check" cells are code and stay verbatim, comments included. After editing, from the lab directory ${lab}:
  cd ${lab} && ${item.python || 'python3'} ${item.builder || 'tools/build_notebooks.py'}        # rebuilds notebooks/ (blanks) and solutions/ from the sources${item.python ? ' (this lab executes its notebooks at build time: use exactly this interpreter, whose packages are pinned, so the committed outputs rebuild byte for byte; a build takes a few minutes)' : ''}
  ${NBSRC}                                                   # every heading in a Markdown cell
  cd ${lab} && python3 -m pytest -q -p no:warnings tests/test_notebook_tooling.py   # if the lab has it: a rebuild of unchanged sources is a no-op
  cd ${lab} && python3 tools/run_notebooks.py notebooks --expect-fail   # if the lab has tools/run_notebooks.py: every blank still stops at its first exercise
  ${LINT} <each source file>
Another agent may be editing other sources of the same lab at the same time: if the tooling test reports a stale notebook that is not generated from your sources, rebuild once more and re-run; never edit another agent's sources. "git diff --stat" must show, among your files, only your sources and the notebooks generated from them.`
    case 'nb':
      return `Hand-written notebooks (.ipynb). Only their Markdown cells are yours; code cells, outputs, ids and metadata are not. Use the helper, one notebook at a time:
  ${NBMD} dump ${R}/<notebook> > ${S}/<name>.cells.txt     (edit the text under each "=== cell N id=... ===" marker; keep the markers; the first line of a cell that is a heading stays verbatim)
  ${NBMD} apply ${S}/<name>.cells.txt ${R}/<notebook>
A blank in notebooks/ and its solution in solutions/ with the same file name are twins: when their Markdown cells are identical (compare the two dumps), edit one dump and apply it to both; when they differ, edit each notebook's own dump. After editing: "${INJECT} ${lab}" must print that it changed nothing (then "git diff --stat" shows only Markdown "source" lines in your notebooks), "${LINT} <notebook>" on each, and the lab's notebook runner if it has one ("cd ${lab} && python3 tools/run_notebooks.py solutions", optional when it takes more than ten minutes; say so in checks_run).`
    case 'builder':
      return `A notebook builder with the Markdown prose in Python string literals (${item.builder}). Edit only the Markdown strings (the notebook intro texts and the ".md(...)" cells); the notebook titles (H1) stay verbatim; the code templates, the SETUP strings, the builder's own code and everything that becomes a code cell stay verbatim. Keep the file valid Python (python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" <file>). Then rebuild and restore the recorded outputs, from ${lab}:
  cd ${lab} && python3 ${item.builder.split('/').pop()}
  cd ${lab} && ${NBOUT} --from-git ${BASE} solutions/*.ipynb      # the code cells did not change, so the committed outputs still apply
  cd ${lab} && python3 -m pytest -q -p no:warnings            # the lab's tests, the notebook tooling test included
  ${LINT} ${lab}/notebooks
"git diff --stat" must show, among your files, only the builder and the notebooks it generates, and a solution's diff must show Markdown source lines only.`
    default:
      return ''
  }
}

const fileList = (item) => item.files.map(f => '  - ' + f).join('\n')
const checkList = (item) => (item.tests && item.tests.length ? 'Checks for these files (every one must pass; run each and report its result line):\n' + item.tests.map(t => '  ' + t).join('\n') : 'Checks for these files: the linter and the link check named above.')

const verifyTargets = (item) => {
  const lab = item.lab ? `${R}/${item.lab}` : null
  switch (item.kind) {
    case 'pct':
      return `The rewritten prose is in the percent-format sources listed and in the notebooks the lab's builder generated from them (under ${lab}/notebooks and ${lab}/solutions). Compare each source with "git show ${BASE}:<relative path>" (the Markdown cells are the "# %% [markdown]" blocks). Make sure that no "# %%", "# %% exercise" or "# %% check" cell changed (diff them), that every heading is in a Markdown cell (${NBSRC}), that "cd ${lab} && python3 -m pytest -q -p no:warnings tests/test_notebook_tooling.py" passes, and lint the sources.`
    case 'nb':
      return `The rewritten prose is in the Markdown cells of the notebooks listed. For each, dump the original ("git show ${BASE}:<relative path> > ${S}/<name>.orig.ipynb"; "${NBMD} dump ${S}/<name>.orig.ipynb") and the rewrite ("${NBMD} dump ${R}/<notebook>") and compare. Make sure that "git diff" on each notebook shows Markdown source lines only (no code, outputs, ids or metadata), that "${INJECT} ${lab}" changes nothing, and lint each notebook.`
    case 'builder':
      return `The rewritten prose is in the Markdown strings of ${item.builder} and in the notebooks it generated under ${lab}/notebooks and ${lab}/solutions. Compare the Markdown cells of each generated blank with its original: "git show ${BASE}:<relative path> > ${S}/<name>.orig.ipynb", then "${NBMD} dump" on both. Make sure the solutions kept their outputs (their diffs show Markdown source lines only), that no code cell changed, and that "cd ${lab} && python3 -m pytest -q -p no:warnings" passes.`
    case 'layer-readme':
      return `Compare the file with "git show ${BASE}:<relative path>". The text between the "<!-- colab-links -->" markers must be byte-identical to the original; "${MDLINKS} <file>" gives 0 broken links.`
    default:
      return `Compare each file with "git show ${BASE}:<relative path>". Run "${MDLINKS} <file>" on each: 0 broken links.`
  }
}


const findingsNote = (item) => `Second-pass findings the linter reports on these files now: word ${item.findings.word}, paren-clause ${item.findings['paren-clause']}, ing ${item.findings.ing}, passive ${item.findings.passive}, semicolon ${item.findings.semicolon}.`
const applyPrompt = (item) => `${COMMON}\n\n${APPLY_RULES}\n\nYour files (you own these and only these; no other agent touches them):\n${fileList(item)}\n\nKind of files: ${item.kind}. ${kindNotes(item)}${partNote(item)}\n\n${findingsNote(item)}\n\n${checkList(item)}`
const verifyPrompt = (item, round, prev) => `${COMMON}\n\n${VERIFY_RULES}\n\nFiles under review (round ${round}):\n${fileList(item)}\n\nKind of files: ${item.kind}. ${verifyTargets(item)}${partVerifyNote(item)}\n\n${checkList(item)}\n\nNotes the agent was given for these files:\n${kindNotes(item)}${partNote(item)}\n\n${findingsNote(item)}` +
  (prev ? `\n\nThis is a re-check. The previous round's findings, and what a fixer did with them, are below. Check that each fix is in place and correct, that each rejection is justified, look for new problems the fixes introduced, then read the whole diff once more.\nPrevious findings: ${JSON.stringify(prev.findings)}\nFixer report: ${JSON.stringify(prev.fix)}` : '')
const fixPrompt = (item, verify) => `${COMMON}\n\n${FIX_RULES}\n\nYour files (you own these and only these):\n${fileList(item)}\n\nKind of files: ${item.kind}. ${kindNotes(item)}${partNote(item)}\n\n${checkList(item)}\n\nThe verifier's findings (JSON):\n${JSON.stringify(verify.findings, null, 1)}\n\nThe verifier's summary: ${verify.summary}`

const LINT_SCHEMA = { type: 'object', properties: { errors: { type: 'integer' }, warnings: { type: 'integer' } }, required: ['errors', 'warnings'] }
const STR_LIST = { type: 'array', items: { type: 'string' } }
const REPORT_SCHEMA = {
  type: 'object',
  properties: { files: STR_LIST, lint: LINT_SCHEMA, applied: { type: 'object', properties: { word: { type: 'integer' }, paren_clause: { type: 'integer' }, other: { type: 'integer' } }, required: ['word', 'paren_clause', 'other'] }, kept_warnings: STR_LIST, rule_breaks: STR_LIST, unsure: STR_LIST, checks_run: STR_LIST },
  required: ['files', 'lint', 'applied', 'kept_warnings', 'rule_breaks', 'unsure', 'checks_run'],
}
const FINDINGS_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['pass', 'fix-needed'] },
    hunks_read: { type: 'integer' },
    lint: LINT_SCHEMA,
    findings: { type: 'array', items: { type: 'object', properties: {
      id: { type: 'integer' }, file: { type: 'string' }, severity: { type: 'string', enum: ['blocking', 'major', 'minor'] },
      location: { type: 'string' }, original: { type: 'string' }, rewritten: { type: 'string' }, problem: { type: 'string' }, fix: { type: 'string' },
    }, required: ['id', 'file', 'severity', 'location', 'original', 'rewritten', 'problem', 'fix'] } },
    summary: { type: 'string' },
  },
  required: ['verdict', 'hunks_read', 'lint', 'findings', 'summary'],
}
const FIX_SCHEMA = {
  type: 'object',
  properties: {
    fixed: { type: 'array', items: { type: 'object', properties: { id: { type: 'integer' }, what: { type: 'string' } }, required: ['id', 'what'] } },
    rejected: { type: 'array', items: { type: 'object', properties: { id: { type: 'integer' }, why: { type: 'string' } }, required: ['id', 'why'] } },
    lint: LINT_SCHEMA,
    checks_run: STR_LIST,
  },
  required: ['fixed', 'rejected', 'lint', 'checks_run'],
}

const needsFix = (v) => !v || v.verdict === 'fix-needed' || (v.findings || []).some(f => f.severity !== 'minor')
const counts = (v) => { const c = { blocking: 0, major: 0, minor: 0 }; for (const f of (v && v.findings) || []) c[f.severity] = (c[f.severity] || 0) + 1; return c }
const effortFor = (item) => (item.words >= 8000 || item.findings.word + item.findings['paren-clause'] >= 40) ? 'xhigh' : 'high'

log(`${LAYER}: ${ITEMS.length} items, ${ITEMS.reduce((n, i) => n + i.words, 0)} words, ${ITEMS.reduce((n, i) => n + i.findings.word + i.findings['paren-clause'], 0)} second-pass findings`)

const results = await pipeline(
  ITEMS,
  (item) => (item.after ? donePromise[item.after].then(() => log(`${item.key}: ${item.after} is done, starting`)) : Promise.resolve())
    .then(() => agent(applyPrompt(item), { label: `apply:${item.key}`, phase: 'Apply', model: 'opus', effort: effortFor(item), schema: REPORT_SCHEMA }))
    .then(r => { log(`applied ${item.key}: ${r ? JSON.stringify(r.applied) + ', lint ' + r.lint.errors + '/' + r.lint.warnings : 'no report'}`); return { item, rewrite: r, verify: [], fixes: [] } })
    .catch(e => { doneResolve[item.key](); throw e }),
  (s) => agent(verifyPrompt(s.item, 1, null), { label: `verify:${s.item.key}`, phase: 'Verify', model: 'opus', effort: 'high', schema: FINDINGS_SCHEMA })
    .then(v => { s.verify.push(v); log(`verified ${s.item.key}: ${v ? v.verdict + ' ' + JSON.stringify(counts(v)) : 'no report'}`); return s })
    .catch(e => { doneResolve[s.item.key](); throw e }),
  async (s) => {
    try {
    let round = 0
    while (round < 2 && needsFix(s.verify[s.verify.length - 1])) {
      const v = s.verify[s.verify.length - 1]
      const f = await agent(fixPrompt(s.item, v || { findings: [], summary: 'verifier returned nothing: re-check the files yourself' }),
        { label: `fix${round + 1}:${s.item.key}`, phase: 'Fix', model: 'opus', effort: 'high', schema: FIX_SCHEMA })
      s.fixes.push(f)
      log(`fixed ${s.item.key} (round ${round + 1}): ${f ? f.fixed.length + ' fixed, ' + f.rejected.length + ' rejected, lint ' + f.lint.errors + '/' + f.lint.warnings : 'no report'}`)
      const v2 = await agent(verifyPrompt(s.item, round + 2, { findings: (v && v.findings) || [], fix: f }),
        { label: `reverify${round + 2}:${s.item.key}`, phase: 'Verify', model: 'opus', effort: 'high', schema: FINDINGS_SCHEMA })
      s.verify.push(v2)
      log(`re-verified ${s.item.key} (round ${round + 2}): ${v2 ? v2.verdict + ' ' + JSON.stringify(counts(v2)) : 'no report'}`)
      round++
    }
    return s
    } finally { doneResolve[s.item.key]() }
  },
)

return results.filter(Boolean).map(s => ({
  key: s.item.key,
  kind: s.item.kind,
  words: s.item.words,
  findings_before: s.item.findings,
  apply: s.rewrite ? { lint: s.rewrite.lint, applied: s.rewrite.applied, kept_warnings: s.rewrite.kept_warnings, rule_breaks: s.rewrite.rule_breaks, unsure: s.rewrite.unsure, checks_run: s.rewrite.checks_run } : null,
  rounds: s.verify.map((v, i) => ({ round: i + 1, verdict: v ? v.verdict : 'none', counts: counts(v), hunks_read: v ? v.hunks_read : 0, lint: v ? v.lint : null, summary: v ? v.summary : '' })),
  fixes: s.fixes.map((f, i) => ({ round: i + 1, fixed: f ? f.fixed.length : 0, rejected: f ? f.rejected : [], lint: f ? f.lint : null, checks_run: f ? f.checks_run : [] })),
  final: (() => { const v = s.verify[s.verify.length - 1]; return v ? v.verdict : 'none' })(),
  open_findings: (() => { const v = s.verify[s.verify.length - 1]; return v ? v.findings.filter(f => f.severity !== 'minor').map(f => ({ severity: f.severity, file: f.file.replace(R + '/', ''), location: f.location, problem: f.problem, fix: f.fix })) : [] })(),
  minor_findings: (() => { const v = s.verify[s.verify.length - 1]; return v ? v.findings.filter(f => f.severity === 'minor').length : 0 })(),
}))
