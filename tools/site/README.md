# tools/site — how the guide site is built

The repo is published as a MkDocs Material site at <https://aniryou.github.io/full-stack-agentic-engineer/>.
Nothing on the site is written twice: the layer pages are generated from the repo's own Markdown and notebooks on
every build.

## Preview it locally

```bash
python3 -m pip install -r requirements-site.txt
python3 tools/site/build_site_content.py     # prints one summary line
mkdocs serve                                 # http://127.0.0.1:8000/full-stack-agentic-engineer/
```

`mkdocs build --strict` is what CI runs; it must finish with no warnings. The first build renders every notebook and
takes a few minutes (mkdocs-jupyter caches the result in `.cache/`, gitignored). The generator and the hooks have tests, which CI runs before
the build: `python3 -m pytest tools/site/tests` (pytest is in `requirements-site.txt`).

## What is where

| Path | Written by | Committed? |
|---|---|---|
| `mkdocs.yml` | by hand, except the `nav:` entries between `# nav-layers:start` / `# nav-layers:end` and the `not_in_nav:` patterns between `# not-in-nav:start` / `# not-in-nav:end`, which the generator rewrites | yes |
| `site/index.md`, `site/overrides/`, `site/stylesheets/`, `site/javascripts/`, `site/assets/` | the landing page, theme and the two small scripts, by hand | yes |
| `tools/site/hooks.py` | MkDocs hooks (below) | yes |
| `site/guide/how-to-use.md`, `site/guide/about.md` | by hand | yes |
| `site/guide/{curriculum,compute,colab}.md` | generator, from `CURRICULUM.md`, `COMPUTE.md`, `COLAB.md` | no (`site/.gitignore`) |
| `site/layers/**` | generator: every layer `README.md` (as `index.md`), every `*.md` doc, every notebook, referenced images | no (`site/.gitignore`) |
| `site_build/` | `mkdocs build` | no |

## What the generator does

`build_site_content.py` (standard library only, safe to re-run) copies the content and rewrites relative links: a
link to a document, notebook or folder with a README goes to its page on the site (anchors re-slugified the way
MkDocs does, or dropped when the heading is gone); a link to anything else in the repo (code, Terraform, YAML,
folders without a README) goes to GitHub `blob/main` or `tree/main`. Anchors into notebooks are resolved against the
ids mkdocs-jupyter gives headings (GitHub's and Jupyter's spellings both work); an anchor to a heading that does not
exist is dropped, and the summary line counts it (`SITE_VERBOSE=1` lists every one).

Notebooks: exercise notebooks get an "Open in Colab" button, using the same URL as `tools/gen_colab_index.py`.
Every lab keeps its notebooks in one layout (guarded by `tools/ci/nb_layout.py`): `notebooks/` for what a learner
opens (exercise blanks, lessons, walkthroughs) and `solutions/` for the worked answer to a blank, under the blank's
file name. A notebook in a `solutions/` folder (`is_solution`, kept identical to `tools/gen_colab_index.py`) gets a
"worked answers" line instead, a "(solution)" suffix in the navigation unless its title already says so, and is left
out of search; a folder's notebooks are listed in file-name order. Notebooks are shown as committed, minus the Colab setup cell at the top (it
only runs on Colab); the site never runs them.

Math: authors write TeX the way GitHub renders it, `$...$` inline and `$$` on its own lines for display. The
generator rewrites the inline form to `\(...\)`, the only inline delimiter the site accepts, so dollar amounts
("$20 / $100", "$0.35/h") stay text: a span counts as TeX only when it has no space just inside the dollars, no
digit after the closing one, and looks like TeX (a backslash, `^`, `_`, braces or a single letter). In notebook
cells the result is written with character references (nbconvert's Markdown would otherwise read the underscores
as emphasis); on Markdown pages as plain `\( \)`, which `pymdownx.arithmatex` takes before any other inline rule.
Display blocks pass through: arithmatex on pages, MathJax on notebook pages.

Navigation: one section per folder, titled from the folder's README H1 (the name part before the dash, humanised when
it is just the folder name; "(Mistral)" / "(GCP)" added when a provider variant's title does not say so); primers
are named "<topic> primer"; notebooks are titled from their first H1, numbered like the file. A folder around a
single entry collapses into it. Deploy targets, fixtures, infra, `notebooks_src/` and `client/` folders (`PLUMBING_DIRS`) keep their pages,
reachable from the READMEs that link them, but stay out of the navigation (`not_in_nav:` in `mkdocs.yml`).

Every generated Markdown page gets front matter naming the file it came from (`source_path`, a repo path; the
layers overview gets `source_url`, the repo tree). `site/overrides/main.html` turns it into a "View on GitHub"
link at the top of the page; hand-written pages under `site/guide/` link to their own source, and notebooks carry
the link (beside "Open in Colab") in their first cell.

## What the hooks do

`tools/site/hooks.py` runs inside `mkdocs build`:

- adds a content hash to the URLs of `site/stylesheets/*.css` and `site/javascripts/*.js`, so a deploy never pairs new
  HTML with a cached, older file;
- moves the ~600 KB of JupyterLab and Pygments CSS that mkdocs-jupyter inlines into every notebook page (it has no
  option to link it instead) into one file, `assets/stylesheets/notebook.<hash>.css`, linked from each notebook page;
- removes mkdocs-jupyter's MathJax 2 configuration (inline math between single dollars) and loads
  `site/javascripts/mathjax.js` and MathJax 3.2.2 only on pages whose text has math: `\( \)` inline, `$$ $$` and
  `\[ \]` display, never `$ $`;
- re-points each notebook table-of-contents entry at the heading it names (mkdocs-jupyter builds the TOC from a copy
  with inline code removed, so a heading with `code` in it got a dead link), then reports how many in-page anchors on
  notebook pages still point nowhere;
- leaves solution notebooks out of the search index, and on every notebook page the duplicate copy of each code cell
  (the text behind the copy button), cell outputs, the `In [ ]:` prompts and the "Copied!" notice.
- turns GitHub alerts (`> [!NOTE]`, `[!TIP]`, `[!IMPORTANT]`, `[!WARNING]`, `[!CAUTION]`) in Markdown pages into
  Material admonitions before rendering, so one source reads as a callout on GitHub and on the site; other
  blockquotes are left as they are (the stylesheet styles them as callouts too).

## Markdown extensions and features

`mkdocs.yml` keeps only what the pages use: `attr_list` and `md_in_html` (the landing page's HTML), `tables`, `toc`
with permalinks, `pymdownx.highlight` / `superfences` (code and Mermaid), `admonition` and `pymdownx.details`
(the callouts the hook makes from GitHub alerts), and `pymdownx.arithmatex` with `\( \)` inline (what the
generator makes of `$...$`) and `$$`, `\[ \]` and `\begin{}` display. A bare `$` is always a dollar. Tabs are not
enabled: nothing uses them.

`site/stylesheets/extra.css` is where the reading experience is set: on desktop the page grid may grow to 72rem
(Material's default 61rem leaves 688px for content beside two sidebars, too little for a 7-column table or a
90-character line of code), prose keeps a 38rem measure inside that column, headings are heavier than
Material's so sections stand out between long paragraphs, tables have tighter padding and zebra rows,
blockquotes read as callouts, notebook cells drop the empty `In [ ]:` prompt column, and display maths scrolls in
its own box. `site/javascripts/tables.js` right-aligns numeric columns and wraps "(verify)" so it reads as a tag. `navigation.prune` keeps each page's HTML to the part of the navigation it is in;
`include_source: false` stops publishing a rewritten copy of every notebook that nothing links (pages link GitHub).

When content lands in a layer, nothing here needs editing: re-run the generator and commit the updated `nav:` in
`mkdocs.yml`.

## Publishing

`.github/workflows/pages.yml` builds and deploys on every push to `main` and on manual dispatch; its actions are pinned
to commits. It tries to switch GitHub Pages on by itself (`enablement: true`). If the `configure-pages` step fails
with a permissions error, the owner enables Pages once: **Settings → Pages → Build and deployment → Source: GitHub
Actions**, then re-runs the workflow.
