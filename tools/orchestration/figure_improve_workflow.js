// The figure improvement workflow (2026-10-30, round 3 of the figures): `args = {docs: [...], concurrency}`, one item per
// document (see the generator in the session's scratchpad, make_fig2_args.py: the document, its figure folders, the
// figures near the one-screen limits from figures-manifest.tsv, the long sections without a figure, the tests that pin
// fragments of its prose, and, for a topic with a core, the chart stage). Per document, in order:
//   1. Chart stage (topics with a core): an agent writes <core>/tools/make_figures.py and tests/test_figures.py, draws
//      two to four charts that the core computes (FIGURE-STYLE.md §7) with tools/orchestration/svgchart.py, inserts them
//      into the primer; an adversarial verifier reads every chart against the core and the prose; a fixer; up to two rounds.
//   2. Improve stage (every document): an agent snapshots the document, judges every figure against the reader's
//      checklist (legible, one subject, one direction, the prose's names, a caption that says what to see, consistency,
//      no redundancy, nothing the prose does not say, accessible), repairs what fails, and adds a figure where a long
//      section explains a mechanism without one; an adversarial verifier reads every figure against the snapshot; a
//      fixer; up to two rounds.
// Agents never run git commands that change state and never run the renderer's `check` mode (the manifest is
// regenerated once at integration).
export const meta = {
  name: 'figure-improve',
  description: 'Charts computed by the cores, every figure improved for the reader, figures added where a mechanism has none: draw, verify, fix',
  phases: [
    { title: 'Charts', detail: 'per topic with a core: make_figures.py + test_figures.py, 2 to 4 pinned charts in the primer', model: 'opus' },
    { title: 'Verify charts', detail: 'an adversarial read of every chart against the core and the prose', model: 'opus' },
    { title: 'Fix charts', detail: 'repair the findings, regenerate, re-test', model: 'opus' },
    { title: 'Improve', detail: 'per document: every figure against the reader checklist, figures added to long sections', model: 'opus' },
    { title: 'Verify', detail: 'render and read every figure against the snapshot', model: 'opus' },
    { title: 'Fix', detail: 'repair the findings, re-render, re-check', model: 'opus' },
  ],
}

const R = '/home/user/full-stack-agentic-engineer'
const S = '/tmp/claude-0/-home-user-full-stack-agentic-engineer/54928949-5727-58e1-ab35-871db6a3c519/scratchpad'
const RENDER = `NODE_PATH=${S}/nodemods/node_modules:/usr/local/lib/node_modules_global node ${R}/tools/orchestration/render_figures.js`
const LINT = `python3 ${R}/tools/orchestration/ste_lint.py`
const MDLINKS = `python3 ${R}/tools/orchestration/mdlinks.py`
const DOCS = args.docs
const MAX = args.concurrency || 8

let active = 0
const waiters = []
const acquire = () => new Promise((res) => { if (active < MAX) { active++; res() } else waiters.push(res) })
const release = () => { active--; if (waiters.length) { active++; waiters.shift()() } }

const tmp = (d) => `${S}/fig2/${d.base}`

const COMMON = (d) => `Repository: ${R} (branch claude/asd-ste100-agent-rewrite-hfcxty). Paths below are relative to the repository root; the absolute path is ${R}/<path>. Use absolute paths in every command; when a command must run inside a lab, write "cd <absolute dir> && <command>" in that one command. Never run a git command that changes state (add, commit, stash, checkout, restore, reset, clean, rebase, merge, push): other agents work in this same checkout at the same time, on other files. Never pip install or npm install anything. Never run the renderer's "check" mode (it writes the one shared manifest; the integrator runs it once). Temporary files go under ${tmp(d)}/ (create it). Never put a model name or model identifier in any file.

Read ${R}/tools/orchestration/FIGURE-STYLE.md completely before you start (§1 when a figure earns its place, §2 the three mechanisms, §3 the Mermaid rules with the one-screen rule, §4 the SVG house style, §5 placement and captions, §6 the checks, §7 charts computed by the cores, §8 figures in notebooks). ${d.style === 'ste' ? `Captions and alt texts in this document are in STE: read §2 of ${R}/tools/orchestration/STE100-STYLE.md (the writing rules).` : 'This is a root document: captions and alt texts are in the repository usual style (CONTRIBUTING.md "Writing"), not in STE; the STE linter does not apply to it.'}

The renderer (prerequisites are installed; use exactly this prefix):
  ${RENDER} md <absolute doc.md> <outdir>      renders every Mermaid block of the document to <outdir>/<doc>-m<N>.png and .svg and prints a verdict per block ("ok", or "FAIL" with the reason); exit 0 only when every block is ok
  ${RENDER} svg <absolute file.svg> <out.png>  the same for one SVG file
Look at every PNG with the Read tool (it shows images). The site's and GitHub's column is about 800 px; a wider figure is scaled down to it; the verdict accounts for that.

Document: ${d.doc} (absolute: ${R}/${d.doc}; ${d.words} words; ${d.n_mermaid} Mermaid blocks and ${d.n_svg} SVG figures at the start of this run). Its figure folder(s): ${d.figures_dirs.join(', ')}.`

const testList = (d) => d.tests.length
  ? 'Tests that pin fragments of this document or compute its numbers (every one must pass; run each and report its result line):\n' + d.tests.map((t) => '  ' + t).join('\n')
  : 'Tests that pin fragments of this document: none known; the linter, the link check and the figure tests still apply.'

const nearList = (d) => d.near_blocks.length
  ? 'Figures near the one-screen limits (from figures-manifest.tsv; a Mermaid block number counts the fences of the document from 1; "scale" is how far the figure is shrunk to fit the 800 px column, "shown_h" its height when shown):\n' + d.near_blocks.map((b) => `  ${b.kind} ${b.file || 'block ' + b.block + ' at line ' + b.line}: ${b.size} px, shown at ${b.scale}, ${b.shown_h} px tall`).join('\n')
  : 'Figures near the one-screen limits: none.'

const candList = (d) => d.candidate_sections.length
  ? 'Long sections without a figure (candidates; add one only where a mechanism earns it, see the rules):\n' + d.candidate_sections.map((s) => `  - ${s}`).join('\n')
  : 'Long sections without a figure: none listed; you may still add one where the rules say a mechanism earns it.'

// ----------------------------------------------------------------- stage 1: charts

const CHART_RULES = (d) => `TASK: add charts that the core computes to the document (FIGURE-STYLE.md §7), with the generator and the test that pin them. The prose is in ${d.style === 'ste' ? 'ASD-STE100' : 'the usual style'} and stays byte-identical outside your insertions and the captions of any figure you replace.

The core: ${d.chart.core_dir} (package ${d.chart.package}). Write the generator at ${d.chart.make_figures} and the test at ${d.chart.test_figures}.${d.chart.ci_note ? ' ' + d.chart.ci_note : ''}

Method:
1. Read the whole document. Read the core's modules and its pinned-number tests (tests/test_primer_numbers.py or the tests listed below). List the relationships between two quantities that the prose states with numbers (a table, a worked example, a sentence with two or three points of a curve) and that a function of the core computes.
2. Choose two to four charts. Hints (check each against the core: the function must exist and compute that quantity; when it does not, choose another relationship or make fewer charts, and say so under skipped):
${d.chart.hints.map((h) => '   - ' + h).join('\n')}
   Prefer the relationships the primer spends paragraphs on. An existing hand-drawn SVG that plots the same quantitative relationship (a curve or bars copied from the pinned numbers) is replaced by the computed chart under the same file name, so the reference in the document stays and only its caption changes; a hand-drawn figure that shows a layout, a timeline or a mechanism stays as it is. Never chart a quantity whose value depends on a trained model, a sample that is not pinned by a test, or a measurement.
3. The generator (${d.chart.make_figures}): imports the core; finds the repository root by walking up from its own path to the folder that holds tools/orchestration/svgchart.py and imports svgchart from there (importlib, like tools/ci/tests/test_svgchart.py does, or sys.path); one builder per chart, "def spec_<name>() -> dict", that computes every value with the core's functions from the inputs the primer names (a device preset, a model preset, a price, a constant the primer quotes), never from a number copied out of the prose; the marks are the primer's pinned points, labelled with the number exactly as the primer prints it, so that the reader finds the sentence; a FIGURES table {"<name>.svg": builder}; a main that writes each chart into the figure folder beside the document with svgchart.write_figure(). Axis labels carry the unit the prose uses; a log axis where the prose reasons in decades; two to four series at most; the series named as the prose names them; the note names the function ("computed by <package>.<module>.<function>"). Read the docstring of ${R}/tools/orchestration/svgchart.py for the spec (kinds, axes, styles, marks, legend, note) and run "python3 ${R}/tools/orchestration/svgchart.py demo ${tmp(d)}/demo.svg" once to see a chart.
4. The test (${d.chart.test_figures}): imports the generator (by path), one test per chart that calls svgchart.check_figure(<path>, <builder>), and one test that every file in FIGURES is referenced by the document (the text contains "figures/<name>.svg"). It must run under the lab's CI command ("cd <lab> && python -m pytest -q", or the command named in the note above).
5. Run the generator twice and compare the files (sha1sum): identical bytes. Render each chart with the svg command and look at the PNG: legible labels, nothing overlapping (move the legend, shorten a name, set "anchor" or "dy" on a mark), the axes labelled with units, the pinned points marked. Fix the spec, not the SVG file.
6. Insert each new chart into the document right after the paragraph that states the relationship (never inside a list, a table, a blockquote or a details block; never between a list and its introduction), in the layout of §5: a blank line, ![<the spec title>](figures/<name>.svg), a blank line, an italic caption of one to three sentences${d.style === 'ste' ? ' in STE' : ''} that says what to see, names the quantity and the function, and says that the core computes it and the test pins it, a blank line. A replaced figure keeps its line and gets a caption that fits the chart. At most one chart per section with a heading of its own, and a chart may join a diagram in the same section when they show different things.
7. Checks, all of them, and report each result line:
   - "cd ${R}/${d.chart.core_dir} && python3 -m pytest -q -p no:warnings" (the whole suite, your new test included) and the tests listed below;
   - "git show HEAD:${d.doc} > ${tmp(d)}/orig.md" then ${d.style === 'ste' ? `"${LINT} --no-fail ${tmp(d)}/orig.md" and "${LINT} --no-fail ${R}/${d.doc}": the error count is not higher than the original's and no caption you wrote has a finding (the linter prints path:line);` : 'read your captions once more for plain English;'}
   - "${MDLINKS} ${R}/${d.doc}": 0 broken links;
   - "cd ${R} && python3 -m pytest -q -p no:warnings tools/ci/tests/test_svg_wellformed.py tools/ci/tests/test_figure_links.py";
   - "${RENDER} svg <chart> <png>" for each chart: fits;
   - "git diff HEAD -- ${R}/${d.doc}": only inserted figure lines and captions, and the captions of replaced figures; every other line byte-identical. Read the whole diff once.
8. Report with the structured output: charts (name, file, the relationship, the function, the pinned marks, new or replaced), skipped (hint or relationship, why), files_written, checks_run (each command and its result line), unsure.
Do not edit any file other than the document, the generator, the test and the chart files. Do not touch the other figures of the document (the improve stage follows).`

const CHART_VERIFY_RULES = (d) => `TASK: you are an adversarial verifier of the charts another agent added to the document below (FIGURE-STYLE.md §7): its generator ${d.chart.make_figures}, its test ${d.chart.test_figures}, the chart files and the insertions in the document; the original document is at git HEAD. Assume the work is wrong somewhere and find it; do not praise.
Severity:
 blocking: a value in a chart that the generator copies as a literal instead of computing with the core (read the generator: every series and every mark value must come from a call into ${d.chart.package}, the inputs being the presets and constants the primer names); a mark whose label is not verbatim in the prose; a chart of a trained, sampled or measured quantity; the test missing, not running under the lab's CI command, or failing; the generator not deterministic (run it twice into ${tmp(d)}/regen/ and compare with sha1sum); a chart file that is not what svgchart.render() makes of its own spec (the test checks it; run it); a replaced figure whose numbers differ from the pinned ones or whose caption no longer fits; a line of prose outside the insertions and the replaced captions that changed (diff the document against HEAD, all of it); a chart that does not fit one screen or is not well-formed; a chart inside a list, a table, a blockquote or a details block; ${d.style === 'ste' ? 'a caption with a linter finding or not in STE (a sentence over 25 words, a dash or arrow as a word, an -ing verb, "may", "should", "could", a contraction); ' : ''}a chart whose relationship the section does not state; a model name or identifier in a file.
 major: an axis without the unit the prose uses, or a linear axis where the prose reasons in decades; a series or a mark the reader cannot map to the prose; a legend or a label that covers data or overlaps; an unreadable label; a missing note naming the function; a caption that does not say what to see or does not name the function and the test; a hand-drawn figure of the same relationship left beside the chart (a duplicate), or a layout figure replaced by a chart; fewer than two charts with no reason under skipped, or a skipped hint whose reason is wrong (the function exists); a test that would pass with an empty FIGURES table or that does not check every chart.
 minor: a style nit, a caption that could be shorter, a tick choice that hides the pinned point.
Method: read the generator and the test; run the core's whole test suite ("cd ${R}/${d.chart.core_dir} && python3 -m pytest -q -p no:warnings") and the tests listed below; regenerate into ${tmp(d)}/regen/ (copy the generator's logic by calling its builders and svgchart.render, or run it with a changed output path if it takes one; never overwrite the committed files) and compare; render every chart (the svg command) and read every PNG; for every mark, find its number in the prose (grep); for every series value, find the function call that computes it; run "git diff HEAD -- ${R}/${d.doc}" and "${MDLINKS} ${R}/${d.doc}"${d.style === 'ste' ? ` and "${LINT} --no-fail ${R}/${d.doc}" against "git show HEAD:${d.doc}"` : ''}. For each finding give the chart or file, the location (the nearest heading and line, or the function), the original, what is wrong and the exact repair. At most 40 findings, most severe first, no duplicates. Count the charts you checked. Do not edit any file.
Verdict: "fix-needed" when any blocking or major finding exists, else "pass".`

const CHART_FIX_RULES = (d) => `TASK: you are the fixer for the charts of the document below. A verifier reported the findings in the JSON at the end. For each finding: read the generator, the test, the chart and the document; when the finding is real, repair it with the smallest change that keeps FIGURE-STYLE.md §7 (fix the spec in the generator and regenerate, never the SVG by hand; keep the prose outside the captions byte-identical); when it is not real, or the repair would break the rules, reject it with a one-line reason. Also repair real problems the verifier missed when you see them. After the fixes: run the generator twice and compare (sha1sum), the core's whole test suite ("cd ${R}/${d.chart.core_dir} && python3 -m pytest -q -p no:warnings"), the tests listed below, "cd ${R} && python3 -m pytest -q -p no:warnings tools/ci/tests/test_svg_wellformed.py tools/ci/tests/test_figure_links.py", "${MDLINKS} ${R}/${d.doc}"${d.style === 'ste' ? `, "${LINT} --no-fail ${R}/${d.doc}"` : ''}, render every chart and look at it, and "git diff HEAD -- ${R}/${d.doc}" once more.
Report with the structured output: fixed (finding id and what you changed), rejected (finding id and why), checks_run (each command and its result line).`

// ----------------------------------------------------------------- stage 2: improve

const IMPROVE_RULES = (d) => `TASK: make every figure of the document as clear as it can be for a reader, and add a figure where a mechanism is explained at length without one. The prose stays byte-identical outside the figures, their captions and alt texts.

Step 0, before any edit, the snapshot the verifier compares against: "mkdir -p ${tmp(d)}/before && cp ${R}/${d.doc} ${tmp(d)}/before/doc.md && cd ${R}/${d.doc.replace(/\/[^/]*$/, '') || '.'} && find . -name '*.svg' -not -path '*/node_modules/*' -exec cp --parents {} ${tmp(d)}/before/ \\;" (the SVG files keep their relative paths under before/).${d.chart ? ' The chart stage already ran on this document: the charts (SVGs with a <desc> that holds a JSON spec, generated by ' + d.chart.make_figures + ') are part of the baseline; never edit a chart file by hand: when its labels, legend or marks need a change, edit the spec in the generator, run it ("cd ' + R + '/' + d.chart.core_dir + ' && python3 ' + d.chart.make_figures.replace(d.chart.core_dir + '/', '') + '" or the file\'s own path) and re-run the core\'s tests.' : ''}

Method:
1. Read the whole document once. Render every Mermaid block (the md command) and every SVG the document references (the svg command); look at every PNG.
2. Judge every figure as a reader who has just read the paragraph before it, against this checklist, and repair what fails (the smallest change that repairs it; a figure that passes is left alone, churn is a defect):
   a. Legible. The labels are readable at the shown scale. A Mermaid figure shown below 0.8 (the list below) gets shorter labels (the name of the part, two or three short lines), a chain wrapped into two rows, or a decision tree re-laid, until it is shown at 0.8 or above, when that is possible without losing a part; when it is not, leave it and say why. No text overlaps or leaves its box (the PNG shows the system font, as a reader's machine does); an SVG label needs 14 px of room.
   b. One subject. The part the section is about stands out: in an SVG the accent box (one accent subject per figure, two when the figure compares two things), a cost or a hazard in the warm box and nothing else warm; in Mermaid the position (first, or the centre) and the shape (§3) carry it.
   c. One direction. Left to right or top to bottom; a return path is a labelled curve, not a second direction; no crossing edges where a re-ordering avoids them; about 10 nodes at most; a decision figure shows every outcome the prose names.
   d. The prose's names. Every label is the name the prose uses (the class, the function, the component), spelled as the prose spells it, and the same part has the same name in every figure of the document.
   e. The caption says what to see. One sentence of alt text that says what the figure shows; a caption of one to three sentences that tells the reader where to look first, what the arrows or the axes mean and which number or part to notice, names the parts as the prose does and cites the section or the function. A caption that only repeats the heading, or an alt text that is a title, is rewritten.${d.style === 'ste' ? ' Captions are in STE: no dash or arrow as a word, no -ing verb, no "may", "should", "could", no contraction, at most 25 words a sentence.' : ''}
   f. Consistency. The SVGs of one document share the box sizes, the font sizes, the stroke widths and a card width (880 to 1000) where the content allows; the same kind of part has the same shape in every Mermaid figure; a figure is not redrawn in the other mechanism for consistency alone.
   g. No redundancy. Two figures that show the same thing are differentiated (one keeps the mechanism, the other its cost or its failure mode) or one is removed with its caption; a figure that restates a table or a list in boxes is removed (§1). Say so in the report.
   h. Meaning. The figure states nothing the prose does not; every number in it appears verbatim in the prose; an arrow the wrong way, a part in the wrong group or a missing decision outcome is corrected against the prose.
   i. Accessible. Colour is never the only carrier of a difference (a dash, a shape or a label carries it too); every figure has an alt text; the text has contrast on the dark card.
3. Add a figure to each candidate section listed below where a mechanism, a flow, a timeline, a layout or a decision is explained over several paragraphs without one, and to any other such section you find, within the density rule of §1 (at most one per section with a heading of its own, a chart beside a diagram allowed). Never a figure for a drill, a catalogue, a table of options, a Verify list, a Q&A or a list. Mermaid, or an SVG when §2 says so (positions, proportions, time on an axis); never a hand-drawn curve (a quantitative relationship is a chart of §7, which the chart stage owns: skip it with that reason). Placement and caption per §5; the figure fits one screen at 0.8 or above where possible.
4. Render again until every block is ok and every SVG fits; look at every PNG once more.
5. Checks, all of them, and report each result line:
   - ${d.style === 'ste' ? `"${LINT} --no-fail ${tmp(d)}/before/doc.md" and "${LINT} --no-fail ${R}/${d.doc}": the error count is not higher than the snapshot's and no caption you wrote has a finding;` : 'read every caption you wrote once more;'}
   - "${MDLINKS} ${R}/${d.doc}": 0 broken links;
   - the tests listed below;
   - "cd ${R} && python3 -m pytest -q -p no:warnings tools/ci/tests/test_svg_wellformed.py tools/ci/tests/test_figure_links.py";
   - "diff ${tmp(d)}/before/doc.md ${R}/${d.doc}": only figure blocks, figure references, captions, alt texts and inserted figures changed; every other line byte-identical. Read the whole diff once.
6. Report with the structured output: figures (one entry per figure of the document: its id (the SVG file name or "block N"), the verdict kept / improved / removed / replaced, and what you changed and why, one line), added (section, kind, file or block, size and scale), skipped (candidate section, why), lint (errors in the snapshot and in your version), checks_run, unsure.
Do not edit any file other than the document and the SVG files it references (and, for a chart, the generator). Do not change the prose outside the captions and alt texts, not even a word.`

const IMPROVE_VERIFY_RULES = (d) => `TASK: you are an adversarial verifier of the figures of the document below. Another agent judged every figure against the reader's checklist (legible, one subject, one direction, the prose's names, a caption that says what to see, consistency, no redundancy, nothing the prose does not say, accessible), repaired what failed and added figures to long sections. The baseline for your comparison is the snapshot it took before editing: ${tmp(d)}/before/doc.md and the SVG files under ${tmp(d)}/before/ (their relative paths). When the snapshot is missing, use "git show HEAD:${d.doc}" and the files at HEAD and report the missing snapshot as a major finding${d.chart ? '; note that the chart stage ran before the snapshot and added computed charts (SVGs whose <desc> holds a JSON spec), which are part of the baseline, not of this agent\'s work' : ''}. Assume the work is wrong somewhere and find it; do not praise.
Severity:
 blocking: a part, an edge, a direction, a decision outcome, an order or a group membership of a baseline figure that is lost or changed without the prose saying so; a part or a fact in a figure or a caption that the prose does not state; a number in a figure that the prose does not print verbatim; a line of prose outside the figures, captions and alt texts that changed (diff the document against the snapshot, all of it); a removed figure whose reason is not a duplicate or a restated table; a figure that no longer fits (the renderer says FAIL) or does not parse; an SVG that is not well-formed, uses an external resource or leaves the house style; a computed chart edited by hand (its file is not what svgchart.render makes of its <desc>: the core's tests/test_figures.py says so); a failing test or check listed below; a figure inside a list, a table, a blockquote or a details block; ${d.style === 'ste' ? 'a caption with a linter finding or not in STE; ' : ''}a model name or identifier in a file.
 major: a checklist item that still fails on a figure: a label unreadable at the shown scale (a Mermaid block shown below 0.8 whose labels could be shorter), text that overlaps or leaves its box, the wrong part highlighted or two warm boxes, two directions or avoidable crossings, a label that is not the name the prose uses, a caption that does not say what to see or only repeats the heading, an alt text that is a title, two figures that show the same thing, colour as the only carrier of a difference; a candidate section with no figure and no reason under skipped, or a wrong reason (the section does explain a mechanism over several paragraphs); a new figure that restates a table or a list, or that its section does not explain; a figure redrawn with no improvement a reader would notice (churn), when the redraw changed what the figure shows.
 minor: a style nit, a caption that could be shorter, a label with a typo, churn that changed only the look.
Method: render the document (the md command) and every SVG it references (the svg command); read every verdict and every PNG, the figures that were not changed included. For each changed, new or removed figure, compare it with the baseline figure and with the prose of its section. Run every check listed below, "${MDLINKS} ${R}/${d.doc}", "cd ${R} && python3 -m pytest -q -p no:warnings tools/ci/tests/test_svg_wellformed.py tools/ci/tests/test_figure_links.py"${d.style === 'ste' ? `, "${LINT} --no-fail ${R}/${d.doc}" against the snapshot's count` : ''} and "diff ${tmp(d)}/before/doc.md ${R}/${d.doc}". For each finding give the figure (file or block), the location (the nearest heading and the line), the original, what is wrong and the exact repair. At most 40 findings, most severe first, no duplicates. Count the figures you checked. Do not edit any file.
Verdict: "fix-needed" when any blocking or major finding exists, else "pass".`

const IMPROVE_FIX_RULES = (d) => `TASK: you are the fixer for the figures of the document below. A verifier reported the findings in the JSON at the end. For each finding: read the current figure, the baseline (${tmp(d)}/before/doc.md and the SVG files under ${tmp(d)}/before/, else git HEAD) and the prose of its section; when the finding is real, repair it with the smallest change that keeps the figure within one screen and the rules of FIGURE-STYLE.md (a computed chart is repaired in its generator and regenerated, never by hand); when it is not real, or the repair would break the rules (a figure that no longer fits, a fact the prose does not state, prose changed outside the figures and captions), reject it with a one-line reason. Also repair real problems the verifier missed when you see them. After the fixes render the document and every SVG until every verdict says ok, look at the PNGs, run every check listed below, "cd ${R} && python3 -m pytest -q -p no:warnings tools/ci/tests/test_svg_wellformed.py tools/ci/tests/test_figure_links.py", "${MDLINKS} ${R}/${d.doc}"${d.style === 'ste' ? `, "${LINT} --no-fail ${R}/${d.doc}"` : ''} and "diff ${tmp(d)}/before/doc.md ${R}/${d.doc}" once more.
Report with the structured output: fixed (finding id and what you changed), rejected (finding id and why), checks_run (each command and its result line).`

// ----------------------------------------------------------------- prompts

const chartPrompt = (d) => `${COMMON(d)}\n\n${CHART_RULES(d)}\n\n${testList(d)}`
const chartVerifyPrompt = (d, report, round, prev) => `${COMMON(d)}\n\n${CHART_VERIFY_RULES(d)}\n\n${testList(d)}\n\nThe chart agent's report (JSON): ${JSON.stringify(report)}\n\nRound ${round}.` +
  (prev ? `\n\nThis is a re-check. The previous round's findings, and what a fixer did with them, are below. Check that each fix is in place and correct, that each rejection is justified, look for new problems the fixes introduced, then re-run the checks and read every chart once more.\nPrevious findings: ${JSON.stringify(prev.findings)}\nFixer report: ${JSON.stringify(prev.fix)}` : '')
const chartFixPrompt = (d, v) => `${COMMON(d)}\n\n${CHART_FIX_RULES(d)}\n\n${testList(d)}\n\nThe verifier's findings (JSON):\n${JSON.stringify(v.findings, null, 1)}\n\nThe verifier's summary: ${v.summary}`

const improvePrompt = (d, chartReport) => `${COMMON(d)}\n\n${IMPROVE_RULES(d)}\n\n${nearList(d)}\n\n${candList(d)}\n\n${testList(d)}` +
  (chartReport ? `\n\nThe chart stage's report for this document (the charts it added or replaced; they are part of your baseline): ${JSON.stringify(chartReport.charts)}` : '')
const improveVerifyPrompt = (d, report, round, prev) => `${COMMON(d)}\n\n${IMPROVE_VERIFY_RULES(d)}\n\n${nearList(d)}\n\n${candList(d)}\n\n${testList(d)}\n\nThe improve agent's report (JSON): ${JSON.stringify(report)}\n\nRound ${round}.` +
  (prev ? `\n\nThis is a re-check. The previous round's findings, and what a fixer did with them, are below. Check that each fix is in place and correct, that each rejection is justified, look for new problems the fixes introduced, then render and read every figure once more.\nPrevious findings: ${JSON.stringify(prev.findings)}\nFixer report: ${JSON.stringify(prev.fix)}` : '')
const improveFixPrompt = (d, v) => `${COMMON(d)}\n\n${IMPROVE_FIX_RULES(d)}\n\n${testList(d)}\n\nThe verifier's findings (JSON):\n${JSON.stringify(v.findings, null, 1)}\n\nThe verifier's summary: ${v.summary}`

// ----------------------------------------------------------------- schemas

const STR_LIST = { type: 'array', items: { type: 'string' } }
const CHART_REPORT_SCHEMA = {
  type: 'object',
  properties: {
    charts: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, file: { type: 'string' }, relationship: { type: 'string' }, function: { type: 'string' }, marks: STR_LIST, status: { type: 'string', enum: ['new', 'replaced'] }, section: { type: 'string' } }, required: ['name', 'file', 'relationship', 'function', 'marks', 'status', 'section'] } },
    skipped: { type: 'array', items: { type: 'object', properties: { what: { type: 'string' }, why: { type: 'string' } }, required: ['what', 'why'] } },
    files_written: STR_LIST, checks_run: STR_LIST, unsure: STR_LIST,
  },
  required: ['charts', 'skipped', 'files_written', 'checks_run', 'unsure'],
}
const IMPROVE_REPORT_SCHEMA = {
  type: 'object',
  properties: {
    figures: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, verdict: { type: 'string', enum: ['kept', 'improved', 'removed', 'replaced'] }, what: { type: 'string' } }, required: ['id', 'verdict', 'what'] } },
    added: { type: 'array', items: { type: 'object', properties: { section: { type: 'string' }, kind: { type: 'string' }, where: { type: 'string' }, size: { type: 'string' } }, required: ['section', 'kind', 'where', 'size'] } },
    skipped: { type: 'array', items: { type: 'object', properties: { section: { type: 'string' }, why: { type: 'string' } }, required: ['section', 'why'] } },
    lint: { type: 'object', properties: { errors_before: { type: 'integer' }, errors_after: { type: 'integer' } }, required: ['errors_before', 'errors_after'] },
    checks_run: STR_LIST, unsure: STR_LIST,
  },
  required: ['figures', 'added', 'skipped', 'lint', 'checks_run', 'unsure'],
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
const bigDoc = (d) => (d.n_mermaid + d.n_svg) >= 8 || d.candidate_sections.length >= 3 || d.near_blocks.length >= 3

// A verify → fix → re-verify loop (up to two fix rounds) for one stage of one document.
async function verifyLoop(d, report, mkVerify, mkFix, phaseV, phaseF, labelV, labelF) {
  const rounds = [], fixes = []
  let v = await agent(mkVerify(d, report, 1, null), { label: `${labelV}:${d.doc}`, phase: phaseV, model: 'opus', effort: 'high', schema: FINDINGS_SCHEMA })
  rounds.push(v)
  log(`${labelV} ${d.doc}: ${v ? v.verdict + ' ' + JSON.stringify(counts(v)) : 'no report'}`)
  let round = 0
  while (round < 2 && needsFix(v)) {
    const f = await agent(mkFix(d, v || { findings: [], summary: 'the verifier returned nothing: re-check the figures yourself' }), { label: `${labelF}${round + 1}:${d.doc}`, phase: phaseF, model: 'opus', effort: 'high', schema: FIX_SCHEMA })
    fixes.push(f)
    log(`${labelF} ${d.doc} (round ${round + 1}): ${f ? f.fixed.length + ' fixed, ' + f.rejected.length + ' rejected' : 'no report'}`)
    v = await agent(mkVerify(d, report, round + 2, { findings: (v && v.findings) || [], fix: f }), { label: `re${labelV}${round + 2}:${d.doc}`, phase: phaseV, model: 'opus', effort: 'high', schema: FINDINGS_SCHEMA })
    rounds.push(v)
    log(`re-${labelV} ${d.doc} (round ${round + 2}): ${v ? v.verdict + ' ' + JSON.stringify(counts(v)) : 'no report'}`)
    round++
  }
  return { rounds, fixes }
}

const summarize = (loop) => ({
  rounds: loop.rounds.map((v, i) => ({ round: i + 1, verdict: v ? v.verdict : 'none', all_fit: v ? v.all_fit : null, figures_checked: v ? v.figures_checked : 0, counts: counts(v), summary: v ? v.summary : '' })),
  fixes: loop.fixes.map((f, i) => ({ round: i + 1, fixed: f ? f.fixed : [], rejected: f ? f.rejected : [], checks_run: f ? f.checks_run : [] })),
  final: (() => { const v = loop.rounds[loop.rounds.length - 1]; return v ? v.verdict : 'none' })(),
  open_findings: (() => { const v = loop.rounds[loop.rounds.length - 1]; return v ? v.findings.filter((f) => f.severity !== 'minor') : [] })(),
  minor_findings: (() => { const v = loop.rounds[loop.rounds.length - 1]; return v ? v.findings.filter((f) => f.severity === 'minor').length : 0 })(),
})

log(`${DOCS.length} documents, ${DOCS.filter((d) => d.chart).length} with a chart stage, ${DOCS.reduce((n, d) => n + d.near_blocks.length, 0)} figures near the limits, ${DOCS.reduce((n, d) => n + d.candidate_sections.length, 0)} candidate sections, ${MAX} documents in flight`)

const results = await parallel(DOCS.map((d) => async () => {
  await acquire()
  const s = { doc: d.doc, chart: null, chartLoop: null, improve: null, improveLoop: null }
  try {
    if (d.chart) {
      s.chart = await agent(chartPrompt(d), { label: `charts:${d.doc}`, phase: 'Charts', model: 'opus', effort: 'xhigh', schema: CHART_REPORT_SCHEMA })
      log(`charts ${d.doc}: ${s.chart ? s.chart.charts.length + ' charts (' + s.chart.charts.filter((c) => c.status === 'replaced').length + ' replaced), ' + s.chart.skipped.length + ' skipped' : 'no report'}`)
      s.chartLoop = await verifyLoop(d, s.chart || { charts: [], skipped: [], note: 'the chart agent returned no report: inspect the generator, the test and the document yourself' }, chartVerifyPrompt, chartFixPrompt, 'Verify charts', 'Fix charts', 'verify-charts', 'fix-charts')
    }
    s.improve = await agent(improvePrompt(d, s.chart), { label: `improve:${d.doc}`, phase: 'Improve', model: 'opus', effort: bigDoc(d) ? 'xhigh' : 'high', schema: IMPROVE_REPORT_SCHEMA })
    log(`improved ${d.doc}: ${s.improve ? s.improve.figures.filter((f) => f.verdict !== 'kept').length + ' of ' + s.improve.figures.length + ' figures changed, ' + s.improve.added.length + ' added, ' + s.improve.skipped.length + ' skipped' : 'no report'}`)
    s.improveLoop = await verifyLoop(d, s.improve || { figures: [], added: [], skipped: [], note: 'the improve agent returned no report: inspect the document yourself' }, improveVerifyPrompt, improveFixPrompt, 'Verify', 'Fix', 'verify', 'fix')
  } finally { release() }
  return s
}))

return results.filter(Boolean).map((s) => ({
  doc: s.doc,
  chart: s.chart ? { charts: s.chart.charts, skipped: s.chart.skipped, files_written: s.chart.files_written, unsure: s.chart.unsure } : null,
  chart_verification: s.chartLoop ? summarize(s.chartLoop) : null,
  improve: s.improve ? { figures: s.improve.figures, added: s.improve.added, skipped: s.improve.skipped, lint: s.improve.lint, unsure: s.improve.unsure } : null,
  improve_verification: s.improveLoop ? summarize(s.improveLoop) : null,
}))
