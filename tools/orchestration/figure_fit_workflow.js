// The one-screen figure workflow: `args = {docs: [{doc, blocks, new_sections, tests}], concurrency}`. One reshaper
// per document makes every Mermaid block of the document fit one screen (FIGURE-STYLE.md §3) and adds a compact
// figure to the named sections; an adversarial verifier renders and reads every figure against the original at
// git HEAD; a fixer repairs the findings; up to two fix rounds. The measurements come from
// tools/orchestration/render_figures.js (`check` writes figures-manifest.tsv, which the CI test reads).
export const meta = {
  name: 'figure-fit',
  description: 'Reshape every figure to fit one screen and add compact figures to long sections: reshape, verify, fix',
  phases: [
    { title: 'Reshape', detail: 'one agent per document: every block fits, candidate sections get a figure', model: 'opus' },
    { title: 'Verify', detail: 'render and read every figure against git HEAD', model: 'opus' },
    { title: 'Fix', detail: 'repair the findings, re-render, re-check', model: 'opus' },
  ],
}

const R = '/home/user/full-stack-agentic-engineer'
const S = '/tmp/claude-0/-home-user-full-stack-agentic-engineer/54928949-5727-58e1-ab35-871db6a3c519/scratchpad'
const RENDER = `NODE_PATH=${S}/nodemods/node_modules:/usr/local/lib/node_modules_global node ${R}/tools/orchestration/render_figures.js`
const LINT = `python3 ${R}/tools/orchestration/ste_lint.py`
const MDLINKS = `python3 ${R}/tools/orchestration/mdlinks.py`
const DOCS = args.docs
const MAX = args.concurrency || 6

// A semaphore: at most MAX documents in flight (each agent launches headless Chromium and runs tests).
let active = 0
const waiters = []
const acquire = () => new Promise((res) => { if (active < MAX) { active++; res() } else waiters.push(res) })
const release = () => { active--; if (waiters.length) { active++; waiters.shift()() } }

const COMMON = `Repository: ${R} (branch claude/asd-ste100-agent-rewrite-hfcxty). Paths below are relative to the repository root; the absolute path is ${R}/<path>. Use absolute paths in every command; when a command must run inside a lab, write "cd <absolute dir> && <command>" in that one command. The original of the document is at git HEAD: "git show HEAD:<relative path>". Never run a git command that changes state (add, commit, stash, checkout, restore, reset, clean, rebase, merge, push): other agents work in this same checkout at the same time, on other files. Never pip install or npm install anything. Temporary files go under ${S}/fig/<your document's base name>/ (create it). Never put a model name or model identifier in any file.

Read ${R}/tools/orchestration/FIGURE-STYLE.md completely before you start: §1 (when a figure earns its place), §2 (Mermaid or SVG), §3 (the Mermaid rules, with the one-screen rule and the sizes that fit), §4 (the SVG house style, with a complete example), §5 (placement and captions) and §6 (the checks). Captions and alt texts are in STE: read §2 of ${R}/tools/orchestration/STE100-STYLE.md (the writing rules).

The renderer (prerequisites are installed; use exactly this prefix):
  ${RENDER} md <absolute doc.md> <outdir>      renders every Mermaid block of the document to <outdir>/<doc>-m<N>.png and .svg and prints a verdict per block: "ok" when the block fits one screen, "FAIL" with the reason (too tall, too wide, or a parse error); exit 0 only when every block is ok
  ${RENDER} svg <absolute file.svg> <out.png>  the same for one SVG file
Look at every PNG with the Read tool (it shows images). The site's and GitHub's column is about 800 px; a wider figure is scaled down to it; the verdict already accounts for that.`

const RESHAPE_RULES = `TASK: make every figure of the document below fit one screen, and add a compact figure to each of the named sections where one earns its place. The prose is in ASD-STE100 and must stay byte-identical outside the figures: your edits are the Mermaid blocks (and any SVG you draw), their captions and alt texts, and the inserted figures with their captions.

Method:
1. Render the document (the md command above) and read the verdicts: the failing blocks are listed below as well. Read the whole document once, so that you know what each figure's section explains.
2. For each failing block, reshape it so that it fits, by FIGURE-STYLE.md §3:
   - Labels name the part (two or three short lines, about 20 characters each, the names the prose uses); the explanation that the old labels carried stays in the prose (it is already there) or goes into the caption, never into a label. Keep a number in a label only when the prose states it verbatim.
   - Orientation: "flowchart LR" for a chain or a decision tree (the questions as diamonds in one row, the outcomes below or beside them), "flowchart TB" only for at most four levels. A sequence diagram: at most 4 participants and about 8 messages, with short notes; drop the messages the prose does not need, or split the sequence into two diagrams (one per phase), each with its own caption. A state diagram: at most 6 states, short transition labels.
   - When the mechanism has two parts, split it into two figures, each with its own caption; when the layout needs columns or proportions, redraw it as an SVG in the house style of §4 (width 720 to 1000, height at most 560, well-formed XML, no external resource) in figures/ beside the document, and reference it with ![alt](figures/name.svg).
   - Keep the mechanism: the same parts, the same edges in the same direction, the same decision outcomes and order, the same subgraph membership. A figure states no fact the prose does not; it never gains a part the old figure did not have.
   - A block that does not parse (the renderer says "parse error"): quote every label that holds parentheses, braces, commas, colons, pipes or quotes, as A["..."], and keep the diagram; then make it fit like the others.
3. For each section named under "add a figure": read the section. When a mechanism, a flow, a timeline, a layout or a decision in it is explained over several paragraphs and no figure shows it, add one figure (Mermaid, or an SVG when §2 says so) right after the paragraph that introduces it (never inside a list, a table, a blockquote or a details block; never between a list and its introduction), that fits one screen, with an alt text and an STE caption of one to three sentences, in the layout of §5 (blank line, figure, blank line, italic caption, blank line). When the section is a catalogue, a table of products or options, a drill, a Q&A or a list, add nothing and say why under skipped. At most one figure per section. The figure uses the section's own names and states no fact the section does not.
4. Render again until every block of the document says ok and the command exits 0. Look at every PNG: nothing overlaps, every label is inside its box, the arrows point the way the prose says, the labels are readable. For an SVG, run the svg command and look at it.
5. Checks, all of them:
   - "git show HEAD:<relative path> > ${S}/fig/<base>/orig.md" then "${LINT} --no-fail ${S}/fig/<base>/orig.md" and "${LINT} --no-fail <absolute doc>": the error count of the document is not higher than the original's, and every caption you wrote has no finding of its own (the linter prints path:line).
   - "${MDLINKS} <absolute doc>": 0 broken links (the image paths resolve).
   - the tests listed below (they pin fragments of the prose): every one passes.
   - "cd ${R} && python3 -m pytest -q tools/ci/tests/test_svg_wellformed.py" when you wrote an SVG.
   - "git diff HEAD -- <absolute doc>": only the figure blocks, their captions and alt texts, and the inserted figures and captions changed; every other line is byte-identical to the original. Read the whole diff once.
6. Report with the structured output: blocks (one entry per reshaped block: block number, line in the original, the size before and after, and what you did: shorter labels, LR, split, SVG), added (section, kind, file or block, size), skipped (section, why), lint (errors in the original and in your version), checks_run (each command and its result line), unsure.
Do not edit any other file than the document and the SVG files you create beside it. Do not touch a figure that already fits unless a split or a new figure needs it. Do not change the prose outside the figures and captions, not even a word.`

const VERIFY_RULES = `TASK: you are an adversarial verifier of the figures of the document below. Another agent reshaped its Mermaid blocks to fit one screen and added figures to some sections; the original is at git HEAD. Assume the work is wrong somewhere and find it; do not praise.
Severity:
 blocking: a figure that does not fit (the renderer says FAIL) or does not parse; a part, an edge, a direction, a decision outcome, an order or a subgraph membership of the original figure that is lost or changed; a part the original figure did not have; a fact in a figure or a caption that the prose of that section does not state; a number in a label that the prose does not state verbatim; a label name that is not the name the prose uses; a line of prose outside the figures, captions and alt texts that changed (diff the document against the original, all of it); a failing test or check listed below; an SVG that is not well-formed, uses an external resource, or is outside the house style (colours, fonts, the card); a caption with a linter finding; a figure inside a list, a table, a blockquote or a details block; a model name or identifier in a file.
 major: a label that a reader cannot map to the prose (cryptic, an abbreviation the prose does not use); text that overlaps or leaves its box; an arrow that points the wrong way; an unreadable label (less than about 11 px when shown); a missing caption or alt text; a caption that is not in STE (a sentence over 25 words, a dash or arrow as a word, an -ing verb, "may", "should", "could", a contraction); a new figure that restates a table or a list (FIGURE-STYLE.md §1) or that the section does not explain; a section named for a figure that got none with no reason under skipped, or whose reason is wrong; two figures that show the same thing.
 minor: a style nit, a caption that could be shorter, a label with a typo, a figure that fits but is near the limit and could be simpler.
Method: render the document (the md command above) and read every verdict and every PNG, the figures that were not changed included. For each changed or new figure, compare it with the original block ("git show HEAD:<relative path>", the block at the same place) and with the prose of its section. Run every check listed below and "git diff HEAD -- <absolute doc>". For each finding give the block or figure, the location (the nearest heading and the line), the original, what is wrong and the exact repair. At most 40 findings, most severe first, no duplicates. Count the figures you checked. Do not edit any file.
Verdict: "fix-needed" when any blocking or major finding exists, else "pass".`

const FIX_RULES = `TASK: you are the fixer for the figures of the document below. A verifier reported the findings in the JSON at the end. For each finding: read the current figure and the original ("git show HEAD:<relative path>"); when the finding is real, repair it with the smallest change that keeps the figure within one screen and the rules of FIGURE-STYLE.md; when it is not real, or the repair would break the rules (a figure that no longer fits, a fact the prose does not state, prose changed outside the figures), reject it with a one-line reason. Also repair real problems the verifier missed when you see them. After the fixes render the document until every block says ok, look at the PNGs, run every check listed below and "git diff HEAD -- <absolute doc>" once more.
Report with the structured output: fixed (finding id and what you changed), rejected (finding id and why), checks_run (each command and its result line).`

const blockList = (d) => d.blocks.length
  ? 'Failing blocks (from tools/orchestration/figures-manifest.tsv; block numbers count the Mermaid fences of the original from 1):\n' + d.blocks.map((b) => `  block ${b.block} at line ${b.line}: ${b.size} px, ${b.verdict}`).join('\n')
  : 'Failing blocks: none (every existing block already fits; your work is the sections below).'
const sectionList = (d) => d.new_sections.length
  ? 'Add a figure (one at most per section, only where one earns its place; see the rules):\n' + d.new_sections.map((s) => `  - "${s}"`).join('\n')
  : 'Add a figure: no section is named for this document.'
const testList = (d) => d.tests.length
  ? 'Tests that pin fragments of this document (every one must pass; run each and report its result line):\n' + d.tests.map((t) => '  ' + t).join('\n')
  : 'Tests that pin fragments of this document: none known; the linter and the link check above still apply.'

const docBlock = (d) => `Document: ${d.doc} (absolute: ${R}/${d.doc}; base name for the temporary folder: ${d.doc.replace(/\//g, '__').replace(/\.md$/, '')})\n\n${blockList(d)}\n\n${sectionList(d)}\n\n${testList(d)}`

const reshapePrompt = (d) => `${COMMON}\n\n${RESHAPE_RULES}\n\n${docBlock(d)}`
const verifyPrompt = (d, round, prev) => `${COMMON}\n\n${VERIFY_RULES}\n\n${docBlock(d)}\n\nRound ${round}.` +
  (prev ? `\n\nThis is a re-check. The previous round's findings, and what a fixer did with them, are below. Check that each fix is in place and correct, that each rejection is justified, look for new problems the fixes introduced, then render and read every figure once more.\nPrevious findings: ${JSON.stringify(prev.findings)}\nFixer report: ${JSON.stringify(prev.fix)}` : '')
const fixPrompt = (d, v) => `${COMMON}\n\n${FIX_RULES}\n\n${docBlock(d)}\n\nThe verifier's findings (JSON):\n${JSON.stringify(v.findings, null, 1)}\n\nThe verifier's summary: ${v.summary}`

const STR_LIST = { type: 'array', items: { type: 'string' } }
const REPORT_SCHEMA = {
  type: 'object',
  properties: {
    blocks: { type: 'array', items: { type: 'object', properties: { block: { type: 'integer' }, line: { type: 'integer' }, before: { type: 'string' }, after: { type: 'string' }, how: { type: 'string' } }, required: ['block', 'line', 'before', 'after', 'how'] } },
    added: { type: 'array', items: { type: 'object', properties: { section: { type: 'string' }, kind: { type: 'string' }, where: { type: 'string' }, size: { type: 'string' } }, required: ['section', 'kind', 'where', 'size'] } },
    skipped: { type: 'array', items: { type: 'object', properties: { section: { type: 'string' }, why: { type: 'string' } }, required: ['section', 'why'] } },
    lint: { type: 'object', properties: { errors_before: { type: 'integer' }, errors_after: { type: 'integer' } }, required: ['errors_before', 'errors_after'] },
    checks_run: STR_LIST, unsure: STR_LIST,
  },
  required: ['blocks', 'added', 'skipped', 'lint', 'checks_run', 'unsure'],
}
const FINDINGS_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['pass', 'fix-needed'] },
    figures_checked: { type: 'integer' },
    all_fit: { type: 'boolean' },
    findings: { type: 'array', items: { type: 'object', properties: {
      id: { type: 'integer' }, severity: { type: 'string', enum: ['blocking', 'major', 'minor'] },
      location: { type: 'string' }, original: { type: 'string' }, problem: { type: 'string' }, fix: { type: 'string' },
    }, required: ['id', 'severity', 'location', 'original', 'problem', 'fix'] } },
    summary: { type: 'string' },
  },
  required: ['verdict', 'figures_checked', 'all_fit', 'findings', 'summary'],
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
const effortFor = (d) => (d.blocks.length + d.new_sections.length >= 4) ? 'xhigh' : 'high'

log(`${DOCS.length} documents, ${DOCS.reduce((n, d) => n + d.blocks.length, 0)} failing blocks, ${DOCS.reduce((n, d) => n + d.new_sections.length, 0)} candidate sections, ${MAX} in flight`)

const results = await parallel(DOCS.map((d) => async () => {
  await acquire()
  const s = { doc: d.doc, reshape: null, verify: [], fixes: [] }
  try {
    s.reshape = await agent(reshapePrompt(d), { label: `reshape:${d.doc}`, phase: 'Reshape', model: 'opus', effort: effortFor(d), schema: REPORT_SCHEMA })
    log(`reshaped ${d.doc}: ${s.reshape ? s.reshape.blocks.length + ' blocks, ' + s.reshape.added.length + ' added, ' + s.reshape.skipped.length + ' skipped' : 'no report'}`)
    let v = await agent(verifyPrompt(d, 1, null), { label: `verify:${d.doc}`, phase: 'Verify', model: 'opus', effort: 'high', schema: FINDINGS_SCHEMA })
    s.verify.push(v)
    log(`verified ${d.doc}: ${v ? v.verdict + ' ' + JSON.stringify(counts(v)) : 'no report'}`)
    let round = 0
    while (round < 2 && needsFix(v)) {
      const f = await agent(fixPrompt(d, v || { findings: [], summary: 'the verifier returned nothing: re-check the figures yourself' }), { label: `fix${round + 1}:${d.doc}`, phase: 'Fix', model: 'opus', effort: 'high', schema: FIX_SCHEMA })
      s.fixes.push(f)
      log(`fixed ${d.doc} (round ${round + 1}): ${f ? f.fixed.length + ' fixed, ' + f.rejected.length + ' rejected' : 'no report'}`)
      v = await agent(verifyPrompt(d, round + 2, { findings: (v && v.findings) || [], fix: f }), { label: `reverify${round + 2}:${d.doc}`, phase: 'Verify', model: 'opus', effort: 'high', schema: FINDINGS_SCHEMA })
      s.verify.push(v)
      log(`re-verified ${d.doc} (round ${round + 2}): ${v ? v.verdict + ' ' + JSON.stringify(counts(v)) : 'no report'}`)
      round++
    }
  } finally { release() }
  return s
}))

return results.filter(Boolean).map((s) => ({
  doc: s.doc,
  reshape: s.reshape ? { blocks: s.reshape.blocks, added: s.reshape.added, skipped: s.reshape.skipped, lint: s.reshape.lint, unsure: s.reshape.unsure, checks_run: s.reshape.checks_run } : null,
  rounds: s.verify.map((v, i) => ({ round: i + 1, verdict: v ? v.verdict : 'none', all_fit: v ? v.all_fit : null, figures_checked: v ? v.figures_checked : 0, counts: counts(v), summary: v ? v.summary : '' })),
  fixes: s.fixes.map((f, i) => ({ round: i + 1, fixed: f ? f.fixed : [], rejected: f ? f.rejected : [], checks_run: f ? f.checks_run : [] })),
  final: (() => { const v = s.verify[s.verify.length - 1]; return v ? v.verdict : 'none' })(),
  open_findings: (() => { const v = s.verify[s.verify.length - 1]; return v ? v.findings.filter((f) => f.severity !== 'minor') : [] })(),
  minor_findings: (() => { const v = s.verify[s.verify.length - 1]; return v ? v.findings.filter((f) => f.severity === 'minor').length : 0 })(),
}))
