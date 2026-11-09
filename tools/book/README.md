# tools/book — the primers as an e-book

`build_book.py` turns the primer-like documents of the repository into one book for offline reading: *The LLM Serving
Stack, from GPUs to Agents* (32 chapters in nine parts, about 320k words, every figure). Notebooks, exercises, labs,
READMEs and code stay out; a link to one of them becomes plain text, a link to another primer becomes a link inside
the book.

Two editions come out of the same chapters:

| Edition | File | For | Figures | Formulas |
|---|---|---|---|---|
| Apple | `the-llm-serving-stack-from-gpus-to-agents.epub` | Apple Books and every EPUB 3 reader | the SVG files as they are; Mermaid diagrams rendered to PNG | MathML |
| Kindle | `…-kindle.epub` and, with Calibre, `…-kindle.azw3` | Send to Kindle (the EPUB) or a Kindle over USB (the AZW3) | every figure a PNG on a light background, for e-ink | text where simple, PNG images otherwise (Kindle renders no MathML) |

```bash
# prerequisites: pandoc 3, node with playwright and mermaid on NODE_PATH (tools/orchestration/render_figures.js says
# how), matplotlib for the Kindle formulas, ebook-convert (Calibre) for the AZW3, epubcheck (pip) to validate
python3 tools/book/build_book.py                      # both editions into build/book/ (gitignored)
python3 tools/book/build_book.py --edition apple      # one edition
python3 tools/book/build_book.py --no-azw3 --no-check # without Calibre and epubcheck
```

What the builder does to each chapter before pandoc sees it (`Build.process_chapter`):

- the H1 becomes the chapter heading under a part heading, every other heading moves one level down, and every
  heading gets a stable id (`ch07-3-why-sparsity`) built from GitHub's anchor of its text, so that the primers'
  cross-links (`../PRIMER.md#3-why-sparsity`) resolve inside the book;
- a link to a notebook, a lab, a README or a source file becomes its text; a web link stays (it needs a connection);
- a Mermaid block becomes a PNG rendered by `tools/orchestration/render_figures.js`, with the figure's caption as alt
  text; an SVG figure is copied (Apple) or rendered to PNG with the house palette swapped for a light one (Kindle,
  `LIGHT_PALETTE`); `svg2png.js` renders many SVGs in one browser;
- a token that looks like an HTML tag but is prose (a quoted `</think>`, a `<pid>` placeholder) is escaped, a bare
  price pair (`$0.70/hr … $5`) is kept from being read as a formula, `\tfrac` becomes `\frac` (the EPUB schema
  rejects the MathML that texmath writes for it);
- in the Kindle edition a formula becomes text (`tex_to_html`: italic variables, sub- and superscripts, Unicode
  symbols, upright `\text{}`) when the transliteration is faithful, else a PNG from matplotlib's mathtext
  (`MathRenderer`); an `aligned` environment becomes one image per row; what mathtext rejects stays as TeX in code.

After pandoc the builder opens the EPUB and checks that every internal link resolves and every image exists, runs
epubcheck when it is installed, and converts the Kindle EPUB with `ebook-convert --output-profile kindle_pw3`.

The book's structure (parts, descriptions, chapter order) is the `BOOK` table at the top of the script; a new primer
is added there. The cover is drawn by `Build.cover` in the figures' house style.
