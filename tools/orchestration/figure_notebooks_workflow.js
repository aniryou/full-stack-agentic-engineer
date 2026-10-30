// Figures in the notebooks (2026-10-30, FIGURE-STYLE.md §8): `args = {labs: [...], catalog, concurrency}`, one item per
// lab with notebooks (the generator in the session's scratchpad, make_nb_args.py: the lab, its kind — percent-source,
// a builder with embedded cells, or hand-written JSON — its notebooks and solution twins, the build command, the tests,
// the documents whose figures fit it first) and the catalogue of every SVG figure a document references (path, kind,
// document, section, alt text, caption). Per lab: an agent gives each notebook at most one figure a primer already
// carries, embedded by its absolute URL on main in the first Markdown cell with a caption that names the source; an
// adversarial verifier reads every embedded figure against the notebook's lesson and checks the twins, the builder
// and the tests; a fixer; up to two rounds. Agents never run git commands that change state.
export const meta = {
  name: 'figure-notebooks',
  description: 'Embed one primer figure per notebook by its absolute URL, with a caption that names the source: embed, verify, fix',
  phases: [
    { title: 'Embed', detail: 'per lab: one figure per notebook at most, blank and solution alike, rebuilt or applied to both twins', model: 'opus' },
    { title: 'Verify', detail: 'every embedded figure against the lesson, the twins, the builder and the tests', model: 'opus' },
    { title: 'Fix', detail: 'repair the findings, rebuild, re-test', model: 'opus' },
  ],
}

const R = '/home/user/full-stack-agentic-engineer'
const S = '/tmp/claude-0/-home-user-full-stack-agentic-engineer/54928949-5727-58e1-ab35-871db6a3c519/scratchpad'
const RAW = 'https://raw.githubusercontent.com/aniryou/full-stack-agentic-engineer/main/'
const LINT = `python3 ${R}/tools/orchestration/ste_lint.py`
const LABS = args.labs
const CATALOG = args.catalog || `${S}/figures-catalog.tsv`
const MAX = (args && args.concurrency) || 2

let active = 0
const waiters = []
const acquire = () => new Promise((res) => { if (active < MAX) { active++; res() } else waiters.push(res) })
const release = () => { active--; if (waiters.length) { active++; waiters.shift()() } }
const tmp = (l) => `${S}/nbfig/${l.base}`

const COMMON = (l) => `Repository: ${R} (branch claude/asd-ste100-agent-rewrite-hfcxty). Paths below are relative to the repository root; the absolute path is ${R}/<path>. Use absolute paths in every command; when a command must run inside a lab, write "cd <absolute dir> && <command>" in that one command. Never run a git command that changes state (add, commit, stash, checkout, restore, reset, clean, rebase, merge, push): other agents work in this same checkout at the same time, on other labs. Never pip install or npm install anything. Temporary files go under ${tmp(l)}/ (create it). Never put a model name or model identifier in any file.

Read §8 of ${R}/tools/orchestration/FIGURE-STYLE.md (figures in notebooks) and §5 (captions) before you start, and §2 of ${R}/tools/orchestration/STE100-STYLE.md (the writing rules; the captions are in STE).

The lab: ${l.lab} (absolute: ${R}/${l.lab}), kind: ${l.kind}. Blank notebooks (notebooks/<name>.ipynb): ${l.notebooks.join(', ')}. Solution twins (solutions/<name>.ipynb): ${l.solutions.length ? l.solutions.join(', ') : 'none'}.${l.build ? `\nThe build command (rebuilds the blanks and the solutions from the sources; a rebuild of unchanged sources is a no-op): ${l.build}` : ''}${l.notes.length ? '\n' + l.notes.map((n) => '- ' + n).join('\n') : ''}

The figure catalogue: ${CATALOG} (tab-separated: path, kind, document, section, alt text, caption; a figure is embedded as ${RAW}<path>). The documents whose figures fit this lab first: ${l.topic_docs.join(', ') || 'none in particular; use the catalogue'}.`

const EMBED_RULES = (l) => `TASK: give each notebook of the lab at most one figure that a primer already carries (FIGURE-STYLE.md §8), so that a reader on Colab, GitHub, JupyterLab or the site sees the mechanism the notebook works through.

Method:
1. Read the catalogue and the lab's README. For each blank notebook, read its first Markdown cell (the title, the tier line, "The one-minute version" or the introduction) and skim the rest to know which mechanism it works through.
2. For each notebook choose at most one figure from the catalogue whose mechanism the notebook works through: first a figure of the documents named above, else any catalogue figure that fits exactly; a chart (kind "chart") that the notebook itself computes or reproduces is the best match; a layout or timeline the notebook walks through is the next. No figure when none fits exactly (say so under skipped with the reason); never a figure for a mechanism the notebook only mentions, never two figures, never a Mermaid block (it cannot be embedded), never a figure drawn for the notebook.
3. Insert the figure into the first Markdown cell, after its first paragraph (the title line and the tier paragraph stay first; the figure goes before "The one-minute version" or the introduction's second paragraph), in this layout: a blank line, ![<the catalogue's alt text>](${RAW}<path>), a blank line, an italic caption *Figure from <the document's title or path as the lab's README names it>, §<section>: <one or two sentences that say what to see, in STE>.*, a blank line. The caption names the document and the section (the catalogue's "section" column, shortened to its number and name) and says what to look at in the figure for this notebook. STE: the imperative or the simple present, active voice, at most 25 words a sentence, no dash or arrow as a word, no -ing verb, no "may", "should", "could", no contraction.
4. Apply it the way the lab is built:
   - percent-source lab: edit the first "# %% [markdown]" block of notebooks_src/<name>.py (each Markdown line starts with "# "), then run the build command; both twins are regenerated from the source;
   - a lab with a builder that embeds the cells: edit the builder's first Markdown cell of each notebook, then run the build command (and the outputs step named above, when named);
   - hand-written: "python3 ${R}/tools/orchestration/nb_md.py dump <blank>" > a file, edit the first cell in the dump, then "python3 ${R}/tools/orchestration/nb_md.py apply <file> <blank> <solution>" (both twins in one command; a blank without a twin alone).
5. Checks, all of them, and report each result line:
   - run the build command a second time (when there is one): the tree does not change ("cd ${R} && git status --porcelain -- ${l.lab}" lists the same files before and after; compare "sha1sum" of every notebook before and after the second run); when there is no build command, "python3 ${R}/tools/inject_colab_bootstrap.py ${R}/${l.lab}" is a no-op (sha1sum before and after);
   - the tests listed below;
   - "cd ${R} && python3 tools/ci/ci.py check" (the notebook layout and the percent-source heading guard; it exits 0);
   - "cd ${R} && python3 -m pytest -q -p no:warnings tools/ci/tests/test_figure_links.py" (every URL names a file in the tree);
   - "${LINT} --no-fail ${R}/${l.lab}/notebooks" and the same for solutions and, for a percent lab, notebooks_src: the error count is not higher than before your edit (run the linter on the originals first: "git show HEAD:<path>" into ${tmp(l)}/orig/), and no caption you wrote has a finding;
   - for each edited notebook, compare its cells with the original ("git show HEAD:<path> > ${tmp(l)}/orig/<name>.ipynb" and a small Python comparison): every cell except the first Markdown cell is byte-identical (code, outputs, ids, metadata), the blank and the solution carry the same first Markdown cell text, and the first cell (the Colab bootstrap, a code cell) is untouched.
6. Report with the structured output: embedded (notebook, figure path, document and section, the caption), skipped (notebook, why), files_changed, checks_run (each command and its result line), unsure.
Do not edit any file outside this lab. Do not change any Markdown outside the inserted lines, and no code cell.`

const VERIFY_RULES = (l) => `TASK: you are an adversarial verifier of the figures another agent embedded in the notebooks of the lab below (FIGURE-STYLE.md §8); the originals are at git HEAD. Assume the work is wrong somewhere and find it; do not praise.
Severity:
 blocking: a URL whose path is not a file in the tree or not a figure of the catalogue; a figure that does not show the mechanism the notebook works through (read the notebook's code cells, not only its introduction); two figures in one notebook; a figure outside the first Markdown cell or inside a list or a blockquote; a change to any cell other than the first Markdown cell of an edited notebook (code, outputs, ids, metadata: compare with "git show HEAD:<path>" cell by cell); the blank and its solution twin with different first cells; the Colab bootstrap cell changed; a rebuild that is not a no-op (run the build command and compare sha1sums), or the injector not a no-op for a hand-written lab; a failing test or check listed below; a caption with a linter finding or not in STE (a sentence over 25 words, a dash or arrow as a word, an -ing verb, "may", "should", "could", a contraction); a caption that names the wrong document or section; a model name or identifier in a file; for a percent lab, a notebook edited directly instead of through notebooks_src (the rebuild would undo it: run it and see).
 major: a notebook with no figure although a catalogue figure of its own topic shows exactly the mechanism it works through (name the figure), or a skipped reason that is wrong; a caption that does not say what to look at for this notebook, or that only repeats the alt text; a figure placed before the tier paragraph or after the introduction's second paragraph; a figure chosen from another topic when the own topic has a closer one.
 minor: a style nit, a caption that could be shorter.
Method: read the agent's report, the catalogue, every edited notebook (blank and twin) and its original; run the build command a second time and compare; run the tests listed below, "cd ${R} && python3 tools/ci/ci.py check", "cd ${R} && python3 -m pytest -q -p no:warnings tools/ci/tests/test_figure_links.py" and "${LINT} --no-fail" over notebooks, solutions and notebooks_src against the originals' counts. For each finding give the notebook, the location, what is wrong and the exact repair. At most 40 findings, most severe first, no duplicates. Count the notebooks you checked. Do not edit any file.
Verdict: "fix-needed" when any blocking or major finding exists, else "pass".`

const FIX_RULES = (l) => `TASK: you are the fixer for the embedded figures of the lab below. A verifier reported the findings in the JSON at the end. For each finding: read the notebook, its original ("git show HEAD:<path>") and the catalogue; when the finding is real, repair it with the smallest change that keeps FIGURE-STYLE.md §8 (through the source or the builder for a built lab, through nb_md.py for a hand-written lab, both twins alike); when it is not real, or the repair would break the rules, reject it with a one-line reason. Also repair real problems the verifier missed when you see them. After the fixes: run the build command twice (when there is one) and compare sha1sums, the tests listed below, "cd ${R} && python3 tools/ci/ci.py check", "cd ${R} && python3 -m pytest -q -p no:warnings tools/ci/tests/test_figure_links.py" and "${LINT} --no-fail" over the lab's notebooks, solutions and notebooks_src.
Report with the structured output: fixed (finding id and what you changed), rejected (finding id and why), checks_run (each command and its result line).`

const testList = (l) => l.tests.length ? 'Tests of this lab that pin the notebooks (every one must pass):\n' + l.tests.map((t) => '  ' + t).join('\n') : 'Tests of this lab that pin the notebooks: none; the build no-op check, ci.py check and the linter still apply.'

const embedPrompt = (l) => `${COMMON(l)}\n\n${EMBED_RULES(l)}\n\n${testList(l)}`
const verifyPrompt = (l, report, round, prev) => `${COMMON(l)}\n\n${VERIFY_RULES(l)}\n\n${testList(l)}\n\nThe embed agent's report (JSON): ${JSON.stringify(report)}\n\nRound ${round}.` +
  (prev ? `\n\nThis is a re-check. The previous round's findings, and what a fixer did with them, are below. Check that each fix is in place and correct, that each rejection is justified, look for new problems the fixes introduced, then re-run the checks.\nPrevious findings: ${JSON.stringify(prev.findings)}\nFixer report: ${JSON.stringify(prev.fix)}` : '')
const fixPrompt = (l, v) => `${COMMON(l)}\n\n${FIX_RULES(l)}\n\n${testList(l)}\n\nThe verifier's findings (JSON):\n${JSON.stringify(v.findings, null, 1)}\n\nThe verifier's summary: ${v.summary}`

const STR_LIST = { type: 'array', items: { type: 'string' } }
const REPORT_SCHEMA = {
  type: 'object',
  properties: {
    embedded: { type: 'array', items: { type: 'object', properties: { notebook: { type: 'string' }, figure: { type: 'string' }, source: { type: 'string' }, caption: { type: 'string' } }, required: ['notebook', 'figure', 'source', 'caption'] } },
    skipped: { type: 'array', items: { type: 'object', properties: { notebook: { type: 'string' }, why: { type: 'string' } }, required: ['notebook', 'why'] } },
    files_changed: STR_LIST, checks_run: STR_LIST, unsure: STR_LIST,
  },
  required: ['embedded', 'skipped', 'files_changed', 'checks_run', 'unsure'],
}
const FINDINGS_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['pass', 'fix-needed'] },
    notebooks_checked: { type: 'integer' },
    findings: { type: 'array', items: { type: 'object', properties: {
      id: { type: 'integer' }, severity: { type: 'string', enum: ['blocking', 'major', 'minor'] },
      location: { type: 'string' }, problem: { type: 'string' }, fix: { type: 'string' },
    }, required: ['id', 'severity', 'location', 'problem', 'fix'] } },
    summary: { type: 'string' },
  },
  required: ['verdict', 'notebooks_checked', 'findings', 'summary'],
}
const FIX_SCHEMA = {
  type: 'object',
  properties: {
    fixed: { type: 'array', items: { type: 'object', properties: { id: { type: 'integer' }, what: { type: 'string' } }, required: ['id', 'what'] } },
    rejected: { type: 'array', items: { type: 'object', properties: { id: { type: 'integer' }, why: { type: 'string' } }, required: ['id', 'why'] } },
    checks_run: STR_LIST,
  },
  required: ['fixed', 'rejected', 'checks_run'],
}

const needsFix = (v) => !v || v.verdict === 'fix-needed' || (v.findings || []).some((f) => f.severity !== 'minor')
const counts = (v) => { const c = { blocking: 0, major: 0, minor: 0 }; for (const f of (v && v.findings) || []) c[f.severity] = (c[f.severity] || 0) + 1; return c }

log(`${LABS.length} labs, ${LABS.reduce((n, l) => n + l.notebooks.length, 0)} blank notebooks, ${MAX} labs in flight`)

const results = await parallel(LABS.map((l) => async () => {
  await acquire()
  const s = { lab: l.lab, embed: null, verify: [], fixes: [] }
  try {
    s.embed = await agent(embedPrompt(l), { label: `embed:${l.lab}`, phase: 'Embed', model: 'opus', effort: l.notebooks.length >= 8 || l.kind !== 'percent' ? 'xhigh' : 'high', schema: REPORT_SCHEMA })
    log(`embedded ${l.lab}: ${s.embed ? s.embed.embedded.length + ' figures, ' + s.embed.skipped.length + ' skipped' : 'no report'}`)
    let v = await agent(verifyPrompt(l, s.embed || { embedded: [], skipped: [], note: 'the embed agent returned no report: inspect the notebooks yourself' }, 1, null), { label: `verify:${l.lab}`, phase: 'Verify', model: 'opus', effort: 'high', schema: FINDINGS_SCHEMA })
    s.verify.push(v)
    log(`verified ${l.lab}: ${v ? v.verdict + ' ' + JSON.stringify(counts(v)) : 'no report'}`)
    let round = 0
    while (round < 2 && needsFix(v)) {
      const f = await agent(fixPrompt(l, v || { findings: [], summary: 'the verifier returned nothing: re-check the notebooks yourself' }), { label: `fix${round + 1}:${l.lab}`, phase: 'Fix', model: 'opus', effort: 'high', schema: FIX_SCHEMA })
      s.fixes.push(f)
      log(`fixed ${l.lab} (round ${round + 1}): ${f ? f.fixed.length + ' fixed, ' + f.rejected.length + ' rejected' : 'no report'}`)
      v = await agent(verifyPrompt(l, s.embed, round + 2, { findings: (v && v.findings) || [], fix: f }), { label: `reverify${round + 2}:${l.lab}`, phase: 'Verify', model: 'opus', effort: 'high', schema: FINDINGS_SCHEMA })
      s.verify.push(v)
      log(`re-verified ${l.lab} (round ${round + 2}): ${v ? v.verdict + ' ' + JSON.stringify(counts(v)) : 'no report'}`)
      round++
    }
  } finally { release() }
  return s
}))

return results.filter(Boolean).map((s) => ({
  lab: s.lab,
  embed: s.embed ? { embedded: s.embed.embedded, skipped: s.embed.skipped, files_changed: s.embed.files_changed, unsure: s.embed.unsure } : null,
  rounds: s.verify.map((v, i) => ({ round: i + 1, verdict: v ? v.verdict : 'none', notebooks_checked: v ? v.notebooks_checked : 0, counts: counts(v), summary: v ? v.summary : '' })),
  fixes: s.fixes.map((f, i) => ({ round: i + 1, fixed: f ? f.fixed : [], rejected: f ? f.rejected : [] })),
  final: (() => { const v = s.verify[s.verify.length - 1]; return v ? v.verdict : 'none' })(),
  open_findings: (() => { const v = s.verify[s.verify.length - 1]; return v ? v.findings.filter((f) => f.severity !== 'minor') : [] })(),
}))
