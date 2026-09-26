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
takes a few minutes.

## What is where

| Path | Written by | Committed? |
|---|---|---|
| `mkdocs.yml` | by hand, except the `nav:` entries between `# nav-layers:start` / `# nav-layers:end`, which the generator rewrites | yes |
| `site/index.md`, `site/overrides/`, `site/stylesheets/`, `site/assets/` | the landing page and theme, by hand | yes |
| `site/guide/how-to-use.md`, `site/guide/about.md` | by hand | yes |
| `site/guide/{curriculum,compute,colab}.md` | generator, from `CURRICULUM.md`, `COMPUTE.md`, `COLAB.md` | no (`site/.gitignore`) |
| `site/layers/**` | generator: every layer `README.md` (as `index.md`), every `*.md` doc, every notebook, referenced images | no (`site/.gitignore`) |
| `site_build/` | `mkdocs build` | no |

## What the generator does

`build_site_content.py` (standard library only, safe to re-run) copies the content and rewrites relative links: a
link to a document, notebook or folder with a README goes to its page on the site (anchors re-slugified the way
MkDocs does, or dropped when the heading is gone); a link to anything else in the repo (code, Terraform, YAML,
folders without a README) goes to GitHub `blob/main` or `tree/main`. Exercise notebooks get an "Open in Colab"
button, using the same URL as `tools/gen_colab_index.py`; notebooks under `solutions/` are grouped under
"Solutions" in the navigation. Notebooks are shown as committed; the site never runs them.

When content lands in a layer, nothing here needs editing: re-run the generator and commit the updated `nav:` in
`mkdocs.yml`.

## Publishing

`.github/workflows/pages.yml` builds and deploys on every push to `main` (except changes under `raw/`) and on manual
dispatch. It tries to switch GitHub Pages on by itself (`enablement: true`). If the `configure-pages` step fails
with a permissions error, the owner enables Pages once: **Settings → Pages → Build and deployment → Source: GitHub
Actions**, then re-runs the workflow.
