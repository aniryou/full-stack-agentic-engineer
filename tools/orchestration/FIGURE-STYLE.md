# FIGURE-STYLE.md — diagrams and images in the primers

The primers explain mechanisms: data paths, sequences in time, layouts in memory, hierarchies, state machines. A
figure shows one of these in ten seconds where the prose takes three paragraphs. This file is the brief for adding
one, the house style, and the checks. It was written for the 2026-10-10 pass that gave every primer its figures
(CLAUDE.md, decisions log) and applies to every figure added since.

## 1. When a figure earns its place

- A figure shows a **mechanism** the prose describes over more than one paragraph: which part talks to which, in
  what order, what lives where, what a step costs. The reader must get the idea from the figure alone and find the
  same names in the prose around it.
- Not a figure: a list of items (a table does that), a single formula, a product list, a decoration, a restatement
  of a table in boxes.
- Density: a primer of about 10k words carries roughly 4 to 8 figures; a short primer 2 to 4; the mental-model
  section ("The one-minute version", "the core idea") almost always one. At most one figure per section (a
  subsection with its own heading counts as a section), except that section; a chart computed by the core (§7) may
  join a diagram in the same section when the two show different things (the mechanism, and how a quantity moves).
- An **ASCII sketch in a code fence stays verbatim** (tests pin some of them; STE treats fences as code). Add a
  figure near it only when the figure shows what the sketch cannot (more parts, time, real structure). Never draw
  the same thing twice.

## 2. Two mechanisms

1. **A Mermaid block** (```` ```mermaid ````) for flows, sequences, state machines, architectures and decision
   trees. GitHub renders it, and so does the site (Material for MkDocs loads mermaid@11); both follow the reader's
   light or dark theme. Text in the diff, easy to review and to correct.
2. **An SVG file** for layouts (memory, blocks, tiles, bits), timelines (steps, batches, pipelines, overlap),
   hierarchies with sizes, and charts. Hand-written, self-contained (no external fonts, images, scripts or
   stylesheets; no base64), in the house style of §4. It lives in `figures/<name>.svg` beside the document that
   embeds it (create the folder); kebab-case, named for what it shows (`decode-step-timeline.svg`). The site
   generator copies every image a layer document references (`tools/site/build_site_content.py`, `collect_images`).

3. **A chart computed by the core** (§7): an SVG in the same house style that `tools/orchestration/svgchart.py`
   renders from values the topic's core computes, with a test that pins it. For a relationship between two
   quantities (a cost against a batch size, a time against a width, a break-even), never drawn by hand.

Pick Mermaid unless the figure needs positions, proportions or time on an axis; Mermaid cannot draw those. Pick a
computed chart for anything with a numeric axis that the core can compute.

## 3. Mermaid rules

- Diagram types: `flowchart LR` or `flowchart TB`, `sequenceDiagram`, `stateDiagram-v2`. Nothing else without a
  reason (`gantt`, `timeline` and `xychart` render differently across versions).
- **No `%%{init: …}%%` directive, no `classDef`, `class`, `style` or `linkStyle` lines.** The site and GitHub
  theme the diagram; hand colours break dark mode. Shapes carry the meaning: `[box]` for a component, `[(cylinder)]`
  for a store, `{diamond}` for a decision, `([pill])` for an entry point, `[[subroutine]]` for a kernel or a job.
- Quote a label that contains parentheses, commas, colons, pipes, braces, quotes or `#`:
  `A["allocate_slots(req, n)"]`. Line breaks inside a label are `<br/>`. Keep edge labels to a few words.
- **One screen.** A reader must see the whole figure without a scroll, with the labels readable. The rule, as
  `render_figures.js` and `tools/ci/tests/test_figures_fit.py` measure it: shown in a content column of 800 px
  (a wider figure is scaled down to it), the figure is at most 600 px tall and scaled down by no more than 0.7
  (so a 16 px label stays about 11 px). A figure that fails is reshaped, not accepted.
- Size, so that a figure fits: at most about 10 nodes and 3 subgraphs; a label of at most three short lines
  (about 20 characters each: the name of the part, not its description, which the prose and the caption give);
  `flowchart LR` for up to five stages in a row and at most three rows of nodes, `flowchart TB` for at most four
  levels; a sequence diagram at most 4 participants and about 8 messages, with short notes; a state diagram at
  most 6 states. A decision tree is wide, not tall: `LR` with the questions as diamonds in one row. Split a
  bigger mechanism into two figures, each with its own caption, or draw it as an SVG (§4) when the layout needs
  columns.
- Label text is plain: no bold, no backticks, no unit explanations; a number in a label appears verbatim in the
  prose.
- Names in a label are the names the prose uses (the class, the function, the component), plain text (no
  backticks in Mermaid labels).

## 4. SVG house style

The two figures in `06-gateway/identity-security/agentic-identity-gcp-lab/docs/` set the style: a dark card that
looks the same in every theme, thin outlined boxes, one teal accent for the subject of the figure, muted edges.

- **Card:** `<svg xmlns="http://www.w3.org/2000/svg" width="W" height="H" viewBox="0 0 W H" role="img"
  aria-label="…" style="color: #E4EAE7">`, a `<title>` equal to the aria-label, then
  `<rect x="0" y="0" width="W" height="H" rx="8" fill="#161E1C"/>`. Width 720 to 1000; height at most 560 (one
  screen, §3: at 1000 wide the figure is shown at 0.8, so up to 700 is possible, but keep to 560 at 800 wide).
- **Text:** `font-family="Bricolage Grotesque, system-ui, sans-serif"`, 12.5 px body, 11 to 11.5 px secondary,
  `font-weight="600"` for a box title; code in `font-family="JetBrains Mono, monospace"` 11 px. Primary text is
  `currentColor` (#E4EAE7); secondary text and edge labels `#97A59F`.
- **Boxes:** `rx="6" fill="none" stroke="currentColor" stroke-width="1.2"`. The **accent** box (the subject of
  the figure): `fill="#16302A" stroke="#4CC5B7" stroke-width="1.4"`, accent text `#7ADACF`. The **warm** box (a
  cost, a bottleneck, a hazard): `fill="#3A2C25" stroke="#E39A62"`. A **group**: `fill="none" stroke="#97A59F"
  stroke-dasharray="5 4"` with a muted title inside its top edge.
- **Edges:** `stroke="#97A59F" stroke-width="1.2"` with an arrow marker; an accent edge `#4CC5B7`. Solid for
  data, dashed (`stroke-dasharray="4 3"`) for control or metadata. Labels beside the edge, never on it.
- **Room for text:** fonts differ per machine, so leave at least 14 px around the longest label in a box, and never
  use `textLength`. Line up boxes on a grid; equal gaps; one arrow direction per figure where possible.
- **Facts:** a number inside a figure appears in the primer verbatim; copy it. A figure states no new fact.
- **Size:** under about 40 KB. Nothing animated.

The template (render it to see the idioms: plain box, accent box, warm box, group, solid and dashed edges, a
labelled return path):

```svg
<svg xmlns="http://www.w3.org/2000/svg" width="880" height="300" viewBox="0 0 880 300" role="img" aria-label="A request enters the API server, the scheduler picks the batch, the model runner executes one step on the GPU, and the KV cache manager owns the blocks." style="color: #E4EAE7">
  <title>A request enters the API server, the scheduler picks the batch, the model runner executes one step on the GPU, and the KV cache manager owns the blocks.</title>
  <rect x="0" y="0" width="880" height="300" rx="8" fill="#161E1C"/>
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#97A59F"/></marker>
    <marker id="arrA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#4CC5B7"/></marker>
  </defs>
  <g font-family="Bricolage Grotesque, system-ui, sans-serif" font-size="12.5" fill="currentColor">
    <rect x="236" y="36" width="420" height="230" rx="8" fill="none" stroke="#97A59F" stroke-width="1" stroke-dasharray="5 4"/>
    <text x="446" y="58" text-anchor="middle" fill="#97A59F">engine process</text>
    <rect x="24" y="110" width="150" height="64" rx="6" fill="none" stroke="currentColor" stroke-width="1.2"/>
    <text x="99" y="136" text-anchor="middle" font-weight="600">API server</text>
    <text x="99" y="156" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="11">POST /v1/chat</text>
    <rect x="266" y="110" width="160" height="64" rx="6" fill="#16302A" stroke="#4CC5B7" stroke-width="1.4"/>
    <text x="346" y="136" text-anchor="middle" font-weight="600">Scheduler</text>
    <text x="346" y="156" text-anchor="middle" fill="#7ADACF" font-size="11.5">picks the batch</text>
    <rect x="476" y="110" width="150" height="64" rx="6" fill="none" stroke="currentColor" stroke-width="1.2"/>
    <text x="551" y="136" text-anchor="middle" font-weight="600">Model runner</text>
    <text x="551" y="156" text-anchor="middle" font-size="11.5" fill="#97A59F">one forward step</text>
    <rect x="266" y="200" width="360" height="46" rx="6" fill="#3A2C25" stroke="#E39A62" stroke-width="1.2"/>
    <text x="446" y="228" text-anchor="middle" font-weight="600">KV cache manager · block pool</text>
    <rect x="706" y="110" width="150" height="64" rx="6" fill="none" stroke="currentColor" stroke-width="1.2"/>
    <text x="781" y="136" text-anchor="middle" font-weight="600">GPU</text>
    <text x="781" y="156" text-anchor="middle" font-size="11.5" fill="#97A59F">HBM · SMs</text>
    <g stroke="#97A59F" stroke-width="1.2" fill="none">
      <path d="M174 142 L264 142" marker-end="url(#arr)"/>
      <path d="M426 142 L474 142" marker-end="url(#arr)"/>
      <path d="M626 142 L704 142" marker-end="url(#arr)"/>
      <path d="M346 174 L346 198" marker-end="url(#arr)" stroke-dasharray="4 3"/>
      <path d="M551 174 L551 198" marker-end="url(#arr)" stroke-dasharray="4 3"/>
    </g>
    <text x="219" y="132" text-anchor="middle" font-size="11" fill="#97A59F">request</text>
    <text x="665" y="132" text-anchor="middle" font-size="11" fill="#97A59F">kernels</text>
    <text x="360" y="190" font-size="11" fill="#97A59F">allocate</text>
    <text x="565" y="190" font-size="11" fill="#97A59F">read / write</text>
    <path d="M781 174 C781 240, 700 272, 99 272 L99 176" fill="none" stroke="#4CC5B7" stroke-width="1.2" marker-end="url(#arrA)"/>
    <text x="440" y="289" text-anchor="middle" font-size="11" fill="#7ADACF">one token per step, streamed back</text>
  </g>
</svg>
```

## 5. Placement and captions

- A figure goes right after the paragraph that introduces what it shows: at the top of a section, or after the
  section's first paragraph. Never inside a list, a table, a blockquote or a `<details>` block, and never between a
  list and its introduction.
- In the Markdown: a blank line, then `![alt text](figures/name.svg)` or the Mermaid fence, a blank line, then an
  italic caption paragraph `*…*`, then a blank line.
- The **alt text** (and the SVG's `<title>`) is one sentence that says what the figure shows. The **caption** is
  one to three sentences: what to see, with the parts named as the prose names them; it can cite a section
  ("§3.2") or a function in code font.
- Captions in a layer document (`00-foundations` to `07-application-agent-framework`) are in STE
  ([`STE100-STYLE.md`](STE100-STYLE.md)): the imperative or the simple present, active voice, at most 25 words a
  sentence, no dash or arrow as a word, no -ing verb, no "may", "should", "could", no contraction, no "e.g.",
  "i.e.", "etc.". `python3 tools/orchestration/ste_lint.py <doc>` must report no more errors than before.
- **A figure is an insertion only.** The existing text, headings, fences, tables, lists, links, callouts and
  mathematics stay verbatim; tests pin fragments of them (`tests/test_primer_numbers.py` and the like), and the
  site navigation depends on the headings.

## 6. Checks before a commit

```bash
# every Mermaid block parses and fits one screen (§3), and a PNG per block to look at (prerequisites in the header)
node tools/orchestration/render_figures.js md <doc.md> <outdir>
# each SVG, to look at, with the same verdict
node tools/orchestration/render_figures.js svg <doc-dir>/figures/<name>.svg <out.png>
# before the commit: every figure of the repository, measured into tools/orchestration/figures-manifest.tsv,
# which CI reads (tools/ci/tests/test_figures_fit.py); commit the manifest with the document
node tools/orchestration/render_figures.js check
python3 tools/orchestration/ste_lint.py <doc.md>        # no new errors (compare with `git stash` or HEAD)
python3 tools/orchestration/mdlinks.py <doc.md>          # the image links resolve
cd <the topic's core>; python3 -m pytest -q tests        # the pinned fragments are still there
```

Look at every rendered figure: nothing overlaps, every label is inside its box, the arrows point the way the
prose says, the figure fits one screen (the script's verdict), and it still reads at the site's column width (about
800 px; a 900 px SVG scales down to it).

For a chart (§7): `cd <core> && python3 tools/make_figures.py` regenerates it, `python3 -m pytest -q tests/test_figures.py`
pins it, and `python3 -m pytest -q tools/ci/tests/test_figure_links.py` (from the root) checks that every figure file is
referenced and that every absolute figure URL in a notebook (§8) names a file in the tree.

## 7. Charts computed by the cores

The primers state relationships between quantities, and the cores compute them: the decode step time against the
batch size, the KV cache against the context length, the bytes a collective moves against its size, the cost of a
student against the tokens it serves, the share of experts a batch touches. The prose gives two or three points of
such a curve and the tests pin them; a chart shows the whole curve. A chart is never drawn by hand: a hand-drawn
curve is a claim nobody checks.

- **Where it lives.** `<core>/tools/make_figures.py` builds a spec per chart (a plain dict: the axes, the series, the
  marks, a note), computes every value with the core's own functions (the inputs are the ones the primer names: a
  datasheet constant, a price, a model preset), renders it with `tools/orchestration/svgchart.py` and writes
  `<topic>/figures/<name>.svg` beside the primer. `<core>/tests/test_figures.py` calls `svgchart.check_figure()` with
  the same spec builders, so the committed file fails the test when the core computes other values or when the file
  is not what the renderer makes of the spec stored in its `<desc>`. Both find the repository root by walking up to
  `tools/orchestration/svgchart.py` (the same way `tests/test_notebook_tooling.py` finds the injector). A topic without
  a core (the capacity-planning primer's `capacity.py`, the FlashAttention deep dive's `fa_calculators.py`, the scaling
  lab's `scalelab`) keeps the two files beside its module.
- **What it shows.** One relationship per chart, two to four series at most, the axes labelled with the unit the
  prose uses (`step time (ms)`, `GB`, `FLOP/byte`), a log axis where the prose reasons in decades. The marks are the
  primer's pinned points, labelled with the number exactly as the primer prints it (`8 sequences: 2.4 ms`), so the
  reader finds the sentence. A `vline` or `hline` names a limit the prose names (a roofline, a break-even, a budget).
  The note names the function that computed the values (`computed by roofline.llm.decode_step_ms`).
- **What it never shows.** A quantity whose value depends on a trained model (distill-core's students and drafts move
  with the CPU, see the 2026-09-27 entry of CLAUDE.md); a number the prose does not reason about; a third axis; a
  fitted curve through measured points (the labs measure, the cores compute).
- **The spec** is documented in the helper's docstring: `kind` (`line` or `bar`), `title` (one sentence: the alt text
  and `<title>`), `x` and `y` (label, `linear` or `log`, optional ticks and bounds), `series` (name, `x`, `y`, a style
  from the house palette: `accent` for the subject, `warm` for a cost or a limit, `plain`, `violet`, `yellow`, `muted`;
  `dash`, `markers`, `step`, `fill`), `marks` (`vline`, `hline`, `point`, `band`, `text`), `legend`, `note`. The default
  card is 880 × 360, within the one-screen rule; the height may grow to 560.
- **In the document** a chart is a figure like any other (§5): `![alt](figures/name.svg)`, an italic caption in STE
  that says what to see, names the quantity and the function, and says that the core computes it and the test pins it
  ("`distillcore.cost.break_even` computes the two lines; `tests/test_figures.py` pins the chart.").

## 8. Figures in notebooks

Colab renders neither a Mermaid block nor a relative image, so a notebook embeds an SVG figure by its absolute URL on
the main branch, which GitHub, Colab, JupyterLab and the site all load:

```markdown
![alt text](https://raw.githubusercontent.com/aniryou/full-stack-agentic-engineer/main/<repo path>/figures/<name>.svg)

*Figure from the topic primer, §3: what to see.*
```

- One figure per notebook at most, in its introduction (the first Markdown cell, after its first paragraph), and only
  a figure a primer already carries (a chart or an SVG; a Mermaid block cannot be embedded). A notebook never gets a
  figure drawn for it alone, and never a figure for a mechanism the notebook does not work through.
- The caption names the document the figure comes from and its section, in the style of that document (STE in the
  layers).
- The blank notebook and its solution twin carry the same cell. In a percent-source lab the edit goes into
  `notebooks_src/<name>.py` and the lab is rebuilt (`python3 tools/build_notebooks.py`); in a hand-written lab the two
  JSON files are edited (`tools/orchestration/nb_md.py` dumps and applies the Markdown cells of a pair).
- `tools/ci/tests/test_figure_links.py` checks that every such URL names a file in the tree; a figure that lands in the
  same pull request as the notebook renders once the pull request merges.

