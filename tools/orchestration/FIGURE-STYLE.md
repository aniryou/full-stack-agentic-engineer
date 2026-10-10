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
  section ("The one-minute version", "the core idea") almost always one. At most one figure per section, except
  that section.
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

Pick Mermaid unless the figure needs positions, proportions or time on an axis; Mermaid cannot draw those.

## 3. Mermaid rules

- Diagram types: `flowchart LR` or `flowchart TB`, `sequenceDiagram`, `stateDiagram-v2`. Nothing else without a
  reason (`gantt`, `timeline` and `xychart` render differently across versions).
- **No `%%{init: …}%%` directive, no `classDef`, `class`, `style` or `linkStyle` lines.** The site and GitHub
  theme the diagram; hand colours break dark mode. Shapes carry the meaning: `[box]` for a component, `[(cylinder)]`
  for a store, `{diamond}` for a decision, `([pill])` for an entry point, `[[subroutine]]` for a kernel or a job.
- Quote a label that contains parentheses, commas, colons, pipes, braces, quotes or `#`:
  `A["allocate_slots(req, n)"]`. Line breaks inside a label are `<br/>`. Keep edge labels to a few words.
- Size: at most about 14 nodes and 3 subgraphs; a sequence diagram at most 5 participants and about 12 messages.
  The site's column is about 700 px wide: use `LR` for up to five stages in a row, `TB` beyond that. Split a
  bigger idea into two figures.
- Names in a label are the names the prose uses (the class, the function, the component), plain text (no
  backticks in Mermaid labels).

## 4. SVG house style

The two figures in `06-gateway/identity-security/agentic-identity-gcp-lab/docs/` set the style: a dark card that
looks the same in every theme, thin outlined boxes, one teal accent for the subject of the figure, muted edges.

- **Card:** `<svg xmlns="http://www.w3.org/2000/svg" width="W" height="H" viewBox="0 0 W H" role="img"
  aria-label="…" style="color: #E4EAE7">`, a `<title>` equal to the aria-label, then
  `<rect x="0" y="0" width="W" height="H" rx="8" fill="#161E1C"/>`. Width 720 to 1000, height as needed.
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
# every Mermaid block parses, and a PNG per block to look at (prerequisites in the script's header)
node tools/orchestration/render_figures.js md <doc.md> <outdir>
# each SVG, to look at
node tools/orchestration/render_figures.js svg <doc-dir>/figures/<name>.svg <out.png>
python3 tools/orchestration/ste_lint.py <doc.md>        # no new errors (compare with `git stash` or HEAD)
python3 tools/orchestration/mdlinks.py <doc.md>          # the image links resolve
cd <the topic's core>; python3 -m pytest -q tests        # the pinned fragments are still there
```

Look at every rendered figure: nothing overlaps, every label is inside its box, the arrows point the way the
prose says, and the figure still reads at the site's column width (about 700 px; a 900 px SVG scales down to it).
