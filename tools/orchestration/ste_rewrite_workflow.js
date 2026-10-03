// The STE-100 rewrite workflow, one layer per run: `args = {layer, items}` as tools/orchestration/ste_items.py builds
// them (disjoint file groups with a kind: md, layer-readme, pct, nb, builder). Each item is rewritten by one agent,
// verified adversarially by a second against `git show HEAD:`, repaired by a third, and re-verified, up to two
// fix rounds. The brief is tools/orchestration/STE100-STYLE.md; the checker tools/orchestration/ste_lint.py.
export const meta = {
  name: 'ste100-rewrite-layer',
  description: 'Rewrite one layer of the repository into ASD-STE100: rewrite, adversarial verify, fix, re-verify',
  phases: [
    { title: 'Rewrite', detail: 'one agent per file group, STE100-STYLE.md as the brief', model: 'opus' },
    { title: 'Verify', detail: 'adversarial fidelity + STE check against git HEAD', model: 'opus' },
    { title: 'Fix', detail: 'verify each finding, repair, re-run the checks', model: 'opus' },
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

const COMMON = `Repository: ${R} (branch claude/asd-ste100-agent-rewrite-hfcxty). File paths below are relative to the repository root; the absolute path is ${R}/<path>. Use absolute paths in every command; when a command must run inside a lab, write "cd <absolute dir> && <command>" in that one command. The original of every prose file is at git HEAD: read it with "git show HEAD:<relative path>". Never run a git command that changes state (add, commit, stash, checkout, restore, reset, clean, rebase, merge, push): many agents work in this same checkout at the same time, on other files. Never pip install anything. Temporary files go under ${S}. Never put a model name or model identifier in any file. The style brief is ${STYLE}; read it completely before you start, in particular section 1 (what does not change, items 11 to 14 included), section 3 (vocabulary: the names of this layer) and section 3.5 (document types).`

const REWRITE_RULES = `TASK: rewrite the prose of the files below into ASD-STE100 (Simplified Technical English), in place, as the brief says.

Target quality: a reader who knows the original must find every fact, number, hedge, cross-reference and claim in your text, in the same order, and nothing new. A reader who knows STE must find short, plain, active sentences (at most 25 words, 20 in an instruction), one meaning per word, the imperative for instructions, no -ing forms outside technical names, no modals, no dashes or arrows in prose, and no loss of precision. Prefer a longer text in STE over a shorter text that loses a nuance. Keep the voice of an engineer who explains the design in a design review. Bold labels that lead a paragraph stay; the prose after them is rewritten. Every heading, fenced block, inline code span, formula, link target, table row and column and number stays exactly as it is. A table cell holds short sentences or a comma list, never a semicolon chain. A technical name of this layer (brief, section 3.2) stays as the topic's primer writes it, and is never a verb.

Method:
1. Read the file in full. Rewrite it section by section with the Edit tool (or Write the whole file when nearly every line changes). On a long file, work in passes of a few sections and lint after each pass.
2. After each file run "${LINT} <file>". Repair every ERROR. Repair every WARN, or keep the flagged word only when it is a technical name, a product name, a quotation or a past participle used as an adjective (a state), and record it in kept_warnings with the reason.
3. Run every check listed for your files (below). A test that reports a missing fragment ("PRIMER.md no longer says: ...") means that fragment must go back verbatim inside an STE sentence; never change a test.
4. Compare your rewrite with the original ("git show HEAD:<path>") paragraph by paragraph: every fact, number, limit, date, (verify) tag, cross-reference and claim present; nothing added; no claim stronger or weaker; every list item and table row still there.
5. Report with the structured output: files (the files you changed), lint (the counts of your last linter run, summed over your files), kept_warnings (word and reason), rule_breaks (a sentence where you kept the meaning and broke an STE rule, with the reason), unsure (anything you could not resolve), checks_run (each command you ran and its result line).
Do not edit any file that is not in your list. Do not touch code cells, fenced code blocks, comments in code, Mermaid diagrams, the Colab cell or Python code outside Markdown cells.`

const VERIFY_RULES = `TASK: you are an adversarial verifier. Another agent rewrote the prose of the files below into ASD-STE100, in place; the originals are at git HEAD ("git show HEAD:<relative path>"). Your job is to find every way the rewrite is wrong. Assume it is wrong somewhere and look for it; do not praise.
Severity:
 blocking: a fact, number, unit, limit, date, (verify) tag, hedge ("roughly", "best-effort", "about"), cross-reference or claim that is lost, changed, added, strengthened or weakened; a heading, link target, fenced block, inline code span, table row or column, list item, quotation or formula that changed; a broken link; a failing test or check listed for the files; a notebook that no longer builds; the reader addressed as a role, an employer or a customer; a model name or identifier in a file; text edited inside a generated section or a code cell.
 major: an STE rule the linter cannot see, broken: a cluster of more than three nouns; a word used with an unapproved meaning (follow = obey, check or test as verbs, close as an adjective, fall = decrease); an -ing word kept that is not a technical name; a passive in an instruction; an unclear "it", "this", "that" or "which"; a technical name used as a verb; two names for one thing, or a name the brief's section 3.4 replaces; a parenthetical aside that is a sentence of its own; a sentence that lost its connector (because, thus, but, if) so that the reasoning is gone; a sentence over 25 words or any other linter ERROR still present; a paragraph of more than six sentences.
 minor: a linter WARN that could be repaired; an awkward or ambiguous sentence; a paragraph whose topic sentence is not first; a list item that no longer parallels its neighbours.
Method: run "${LINT} <file>" on each file and every check listed for the files. Then compare the original with the rewrite paragraph by paragraph, all of it, not a sample. For each finding give the file, the location (the nearest heading and a quote of the rewritten sentence), the original text, the problem and the exact replacement text. At most 60 findings per file, most severe first, no duplicates. Count the sentences you compared. Do not edit any file.
Verdict: "fix-needed" when any blocking or major finding exists, else "pass".`

const FIX_RULES = `TASK: you are the fixer for the STE rewrite of the files below. A verifier reported the findings in the JSON at the end. For each finding: read the current text and the original ("git show HEAD:<path>"); if the finding is real, repair it with the smallest change that keeps the text in STE; if it is not real, or the fix would break section 1 of the brief (facts, code, headings, link targets, structure, quotations, meaning, pinned fragments), reject it with a one-line reason. Also repair real problems the verifier missed when you see them. After the fixes run the linter on each file (0 errors; repair or justify every warning) and every check listed for your files, and re-read each changed paragraph against the original.
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
  cd ${lab} && python3 ${item.builder || 'tools/build_notebooks.py'}        # rebuilds notebooks/ (blanks) and solutions/ from the sources
  ${NBSRC}                                                   # every heading in a Markdown cell
  cd ${lab} && python3 -m pytest -q -p no:warnings tests/test_notebook_tooling.py
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
  cd ${lab} && ${NBOUT} --from-git HEAD solutions/*.ipynb      # the code cells did not change, so the committed outputs still apply
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
      return `The rewritten prose is in the percent-format sources listed and in the notebooks the lab's builder generated from them (under ${lab}/notebooks and ${lab}/solutions). Compare each source with "git show HEAD:<relative path>" (the Markdown cells are the "# %% [markdown]" blocks). Make sure that no "# %%", "# %% exercise" or "# %% check" cell changed (diff them), that every heading is in a Markdown cell (${NBSRC}), that "cd ${lab} && python3 -m pytest -q -p no:warnings tests/test_notebook_tooling.py" passes, and lint the sources.`
    case 'nb':
      return `The rewritten prose is in the Markdown cells of the notebooks listed. For each, dump the original ("git show HEAD:<relative path> > ${S}/<name>.orig.ipynb"; "${NBMD} dump ${S}/<name>.orig.ipynb") and the rewrite ("${NBMD} dump ${R}/<notebook>") and compare. Make sure that "git diff" on each notebook shows Markdown source lines only (no code, outputs, ids or metadata), that "${INJECT} ${lab}" changes nothing, and lint each notebook.`
    case 'builder':
      return `The rewritten prose is in the Markdown strings of ${item.builder} and in the notebooks it generated under ${lab}/notebooks and ${lab}/solutions. Compare the Markdown cells of each generated blank with its original: "git show HEAD:<relative path> > ${S}/<name>.orig.ipynb", then "${NBMD} dump" on both. Make sure the solutions kept their outputs (their diffs show Markdown source lines only), that no code cell changed, and that "cd ${lab} && python3 -m pytest -q -p no:warnings" passes.`
    case 'layer-readme':
      return `Compare the file with "git show HEAD:<relative path>". The text between the "<!-- colab-links -->" markers must be byte-identical to the original; "${MDLINKS} <file>" gives 0 broken links.`
    default:
      return `Compare each file with "git show HEAD:<relative path>". Run "${MDLINKS} <file>" on each: 0 broken links.`
  }
}

const rewritePrompt = (item) => `${COMMON}\n\n${REWRITE_RULES}\n\nYour files (you own these and only these; no other agent touches them):\n${fileList(item)}\n\nKind of files: ${item.kind}. ${kindNotes(item)}\n\n${checkList(item)}`
const verifyPrompt = (item, round, prev) => `${COMMON}\n\n${VERIFY_RULES}\n\nFiles under review (round ${round}):\n${fileList(item)}\n\nKind of files: ${item.kind}. ${verifyTargets(item)}\n\n${checkList(item)}\n\nNotes the rewriter was given for these files:\n${kindNotes(item)}` +
  (prev ? `\n\nThis is a re-check. The previous round's findings, and what a fixer did with them, are below. Check that each fix is in place and correct, that each rejection is justified, look for new problems the fixes introduced, then re-read the whole text once more against the original.\nPrevious findings: ${JSON.stringify(prev.findings)}\nFixer report: ${JSON.stringify(prev.fix)}` : '')
const fixPrompt = (item, verify) => `${COMMON}\n\n${FIX_RULES}\n\nYour files (you own these and only these):\n${fileList(item)}\n\nKind of files: ${item.kind}. ${kindNotes(item)}\n\n${checkList(item)}\n\nThe verifier's findings (JSON):\n${JSON.stringify(verify.findings, null, 1)}\n\nThe verifier's summary: ${verify.summary}`

const LINT_SCHEMA = { type: 'object', properties: { errors: { type: 'integer' }, warnings: { type: 'integer' } }, required: ['errors', 'warnings'] }
const STR_LIST = { type: 'array', items: { type: 'string' } }
const REPORT_SCHEMA = {
  type: 'object',
  properties: { files: STR_LIST, lint: LINT_SCHEMA, kept_warnings: STR_LIST, rule_breaks: STR_LIST, unsure: STR_LIST, checks_run: STR_LIST },
  required: ['files', 'lint', 'kept_warnings', 'rule_breaks', 'unsure', 'checks_run'],
}
const FINDINGS_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['pass', 'fix-needed'] },
    sentences_compared: { type: 'integer' },
    lint: LINT_SCHEMA,
    findings: { type: 'array', items: { type: 'object', properties: {
      id: { type: 'integer' }, file: { type: 'string' }, severity: { type: 'string', enum: ['blocking', 'major', 'minor'] },
      location: { type: 'string' }, original: { type: 'string' }, rewritten: { type: 'string' }, problem: { type: 'string' }, fix: { type: 'string' },
    }, required: ['id', 'file', 'severity', 'location', 'original', 'rewritten', 'problem', 'fix'] } },
    summary: { type: 'string' },
  },
  required: ['verdict', 'sentences_compared', 'lint', 'findings', 'summary'],
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
const effortFor = (item) => (item.kind === 'md' && item.words >= 8000) ? 'xhigh' : 'high'

log(`${LAYER}: ${ITEMS.length} items, ${ITEMS.reduce((n, i) => n + i.words, 0)} words`)

const results = await pipeline(
  ITEMS,
  (item) => agent(rewritePrompt(item), { label: `rewrite:${item.key}`, phase: 'Rewrite', model: 'opus', effort: effortFor(item), schema: REPORT_SCHEMA })
    .then(r => { log(`rewrote ${item.key}: lint ${r ? r.lint.errors + ' errors / ' + r.lint.warnings + ' warnings' : 'no report'}`); return { item, rewrite: r, verify: [], fixes: [] } }),
  (s) => agent(verifyPrompt(s.item, 1, null), { label: `verify:${s.item.key}`, phase: 'Verify', model: 'opus', effort: 'high', schema: FINDINGS_SCHEMA })
    .then(v => { s.verify.push(v); log(`verified ${s.item.key}: ${v ? v.verdict + ' ' + JSON.stringify(counts(v)) : 'no report'}`); return s }),
  async (s) => {
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
  },
)

return results.filter(Boolean).map(s => ({
  key: s.item.key,
  kind: s.item.kind,
  words: s.item.words,
  rewrite: s.rewrite ? { lint: s.rewrite.lint, kept_warnings: s.rewrite.kept_warnings, rule_breaks: s.rewrite.rule_breaks, unsure: s.rewrite.unsure, checks_run: s.rewrite.checks_run } : null,
  rounds: s.verify.map((v, i) => ({ round: i + 1, verdict: v ? v.verdict : 'none', counts: counts(v), sentences_compared: v ? v.sentences_compared : 0, lint: v ? v.lint : null, summary: v ? v.summary : '' })),
  fixes: s.fixes.map((f, i) => ({ round: i + 1, fixed: f ? f.fixed.length : 0, rejected: f ? f.rejected : [], lint: f ? f.lint : null, checks_run: f ? f.checks_run : [] })),
  final: (() => { const v = s.verify[s.verify.length - 1]; return v ? v.verdict : 'none' })(),
  open_findings: (() => { const v = s.verify[s.verify.length - 1]; return v ? v.findings.filter(f => f.severity !== 'minor').map(f => ({ severity: f.severity, file: f.file.replace(R + '/', ''), location: f.location, problem: f.problem, fix: f.fix })) : [] })(),
  minor_findings: (() => { const v = s.verify[s.verify.length - 1]; return v ? v.findings.filter(f => f.severity === 'minor').length : 0 })(),
}))
