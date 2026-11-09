#!/usr/bin/env python3
"""build_book.py: the primers of this repository as an e-book, for reading offline.

Two editions from the same chapters:

* **apple** — an EPUB 3 for Apple Books (and any EPUB reader): the SVG figures as they are, Mermaid diagrams rendered to
  PNG, formulas as MathML.
* **kindle** — an EPUB for Send to Kindle plus, when Calibre's ``ebook-convert`` is installed, an AZW3 to copy onto a
  Kindle: every figure as a PNG on a light background (e-ink), formulas as text where they are simple and as PNG
  images otherwise (Kindle devices render no MathML).

The chapters are the primer-like documents (``BOOK`` below), in the order of the stack: Foundations, then the layers
from the hardware up to the agents. Notebooks, exercises, labs, READMEs and code stay out. Each chapter is processed
before pandoc sees it: its headings move one level down under a part heading and get stable ids; a link to another
primer becomes a link inside the book, a link to a notebook, a lab or a repository file becomes plain text, a web link
stays; Mermaid blocks become images (``tools/orchestration/render_figures.js``); the captions and the prose stay as they
are.

    python3 tools/book/build_book.py [--out DIR] [--edition apple|kindle|both] [--no-azw3] [--no-check]

Needs: pandoc (3.x), node with playwright and mermaid on NODE_PATH (the renderer's prerequisites), and for the Kindle
edition matplotlib (with Pillow). ``ebook-convert`` (Calibre) makes the AZW3; ``epubcheck`` (the PyPI package, needs
Java) validates the EPUBs when installed. Standard library otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
RENDER = ROOT / "tools" / "orchestration" / "render_figures.js"
SVG2PNG = HERE / "svg2png.js"
REPO_URL = "https://github.com/aniryou/full-stack-agentic-engineer"
SITE_URL = "https://aniryou.github.io/full-stack-agentic-engineer/"

TITLE = "The LLM Serving Stack, from GPUs to Agents"
SUBTITLE = "The primers of full-stack-agentic-engineer"
AUTHOR = "Anil"

# ------------------------------------------------------------------ the book: parts and chapters, in reading order
BOOK = [
    ("Foundations", "The model itself, below the stack: how a transformer computes, how much memory and time a "
     "deployment needs, which open-weight models exist, and the three ideas that change serving the most: sparsity, "
     "reinforcement learning with thinking, and distillation.", [
        "00-foundations/transformers/docs/transformer-primer.md",
        "00-foundations/gpu-capacity-planning/PRIMER.md",
        "00-foundations/model-landscape/open-weight-llms-primer.md",
        "00-foundations/mixture-of-experts/PRIMER.md",
        "00-foundations/rl-and-thinking-models/PRIMER.md",
        "00-foundations/distillation/PRIMER.md",
    ]),
    ("Hardware and fabric", "The physical layer: GPUs and their memory, the roofline, NVLink and the network fabrics, "
     "storage, cold start, reliability and the cost of a token.", [
        "01-hardware-gpu-fabric/gpu-primer/gpu-primer.md",
        "01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md",
        "01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md",
    ]),
    ("CUDA, NCCL and runtime", "The software substrate of a GPU: the execution model, memory access patterns, "
     "streams and graphs, collectives, how a container gets a GPU, how GPUs are shared and watched.", [
        "02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md",
    ]),
    ("Kubernetes and GPU scheduling", "How a GPU becomes schedulable, how the scheduler places pods on it, gangs, "
     "topology, queues and quotas, and how a cluster gets capacity.", [
        "03-kubernetes-gpu/gpu-scheduling/PRIMER.md",
    ]),
    ("Inference engine", "Running one model fast: the KV cache and paged attention, FlashAttention, the step loop of an "
     "engine with batching, caching and speculation, quantization, and vLLM at the source level.", [
        "04-inference-engine/kv-cache/kv-cache-primer.md",
        "04-inference-engine/paged-attention/paged-attention-primer.md",
        "04-inference-engine/flash-attention/flash-attention-primer.md",
        "04-inference-engine/serving-engine/PRIMER.md",
        "04-inference-engine/quantization/PRIMER.md",
        "04-inference-engine/vllm-internals/vllm-internals-primer.md",
        "04-inference-engine/flash-attention/flash-attention-deep-dive.md",
    ]),
    ("Orchestrator", "Many engine replicas as one service: routing signals, flow control, autoscaling, prefill/decode "
     "disaggregation, KV cache beyond HBM, and the Kubernetes-native stack.", [
        "05-orchestrator/serving-orchestration/PRIMER.md",
    ]),
    ("Gateway", "The control plane in front of the models: identity and delegation for agents, scaling, admission "
     "and cost, and one front door for many models.", [
        "06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md",
        "06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md",
        "06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/02-reference-architecture.md",
        "06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/scaling-agentic-solutions-on-google-cloud.md",
        "06-gateway/llm-gateway/PRIMER.md",
    ]),
    ("Agents and applications", "The agent: durable, long-running workflows, retrieval with embeddings and vector "
     "databases, sandboxed execution of model-generated code, and memory.", [
        "07-application-agent-framework/long-running-durable/PRIMER.md",
        "07-application-agent-framework/long-running-durable/lra-core/PRIMER.md",
        "07-application-agent-framework/long-running-durable/lra-gcp/docs/primer.md",
        "07-application-agent-framework/retrieval-rag/embeddings-lab/docs/primer.md",
        "07-application-agent-framework/retrieval-rag/vector-databases-primer.md",
        "07-application-agent-framework/sandboxed-execution/PRIMER.md",
        "07-application-agent-framework/agent-memory/PRIMER.md",
    ]),
    ("Appendix: the same design on another provider", "A primer written for one provider, worked again for "
     "another: the concepts are the same, the product names and the numbers differ.", [
        "06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/mistral/01-scaling-primer.md",
    ]),
]

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]

# ------------------------------------------------------------------ Markdown helpers

FENCE_RE = re.compile(r"^(?P<indent>[ \t]{0,3})(?P<fence>```+|~~~+)(?P<info>[^\n]*)$")


def split_fences(text: str) -> list[tuple[bool, str]]:
    """[(is_code, chunk)] where code chunks are fenced blocks (fence lines included)."""
    out: list[tuple[bool, str]] = []
    buf: list[str] = []
    in_fence = None
    for line in text.split("\n"):
        m = FENCE_RE.match(line)
        if in_fence is None:
            if m:
                if buf:
                    out.append((False, "\n".join(buf)))
                    buf = []
                in_fence = m.group("fence")[0] * len(m.group("fence"))
                buf.append(line)
            else:
                buf.append(line)
        else:
            buf.append(line)
            if m and m.group("fence")[0] == in_fence[0] and len(m.group("fence")) >= len(in_fence) and not m.group("info").strip():
                out.append((True, "\n".join(buf)))
                buf = []
                in_fence = None
    if buf:
        out.append((in_fence is not None, "\n".join(buf)))
    return out


def protect_inline_code(text: str) -> tuple[str, list[str]]:
    """Replace inline code spans with placeholders so that regexes never touch their content."""
    spans: list[str] = []

    def keep(m: re.Match) -> str:
        spans.append(m.group(0))
        return f"\x00{len(spans) - 1}\x00"

    return re.sub(r"(`+)(?!`)(.+?)(?<!`)\1(?!`)", keep, text), spans


def restore_inline_code(text: str, spans: list[str]) -> str:
    return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], text)


def gh_slug(heading: str) -> str:
    """GitHub's anchor for a heading (the anchors the primers link to)."""
    t = re.sub(r"`([^`]*)`", r"\1", heading)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"[*_]{1,3}(?=\S)(.+?)(?<=\S)[*_]{1,3}", r"\1", t)
    t = t.strip().lower()
    t = re.sub(r"[^\w\- ]", "", t)
    return t.replace(" ", "-")


def looks_like_price(tex: str) -> bool:
    """$0.70/hr … $: a dollar pair that pandoc would read as a formula but that holds no TeX."""
    return bool(re.match(r"\d", tex)) and not re.search(r"[\\^_{}=]", tex)


def protect_math(text: str) -> tuple[str, list[str]]:
    maths: list[str] = []

    def keep_display(m: re.Match) -> str:
        maths.append("$$" + m.group(1).replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac") + "$$")
        return f"\x01{len(maths) - 1}\x01"

    def keep_inline(m: re.Match) -> str:
        tex = m.group(1)
        if looks_like_price(tex):
            return "\\$" + tex + "\\$"
        maths.append("$" + tex.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac") + "$")
        return f"\x01{len(maths) - 1}\x01"

    text = DISPLAY_MATH_RE.sub(keep_display, text)
    text = INLINE_MATH_RE.sub(keep_inline, text)
    text = re.sub(r"(?<!\\)\$(?=\d)", r"\\$", text)   # a price that is left: never a formula
    return text, maths


def restore_math(text: str, maths: list[str]) -> str:
    return re.sub(r"\x01(\d+)\x01", lambda m: maths[int(m.group(1))], text)


HTML_TAGS = {"br", "sub", "sup", "b", "i", "em", "strong", "code", "kbd", "span", "a", "img", "details", "summary", "u", "s",
             "del", "ins", "small", "mark", "abbr", "p", "div", "table", "thead", "tbody", "tr", "td", "th", "ul", "ol", "li",
             "blockquote", "pre", "hr", "h1", "h2", "h3", "h4", "h5", "h6", "dl", "dt", "dd", "figure", "figcaption", "cite", "q"}
STRAY_TAG_RE = re.compile(r"<(/?)([A-Za-z][A-Za-z0-9-]*)(?![:\w@.-])")


def escape_stray_tags(text: str) -> str:
    """A token that looks like an HTML tag but is not one (a quoted </think>, a <pid> placeholder) is prose: pandoc
    would pass it through as raw HTML and the XHTML would not be well-formed. Autolinks (<https://...>) stay."""
    def fix(m: re.Match) -> str:
        if m.group(2).lower() in HTML_TAGS:
            return m.group(0)
        return "&lt;" + m.group(1) + m.group(2)
    return STRAY_TAG_RE.sub(fix, text)


HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
IMG_RE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<src>[^)\s]+)\)[ \t]*$")
LINK_RE = re.compile(r"(?<!!)\[(?P<text>[^\]]*)\]\((?P<url>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
INLINE_MATH_RE = re.compile(r"(?<![\\$\w])\$(?=[^\s$])((?:\\.|[^$\n])+?)(?<![\s\\])\$(?![\d$])")
DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.S)


class Chapter:
    def __init__(self, idx: int, path: str, part_no: int):
        self.idx = idx
        self.path = path
        self.part_no = part_no
        self.cid = f"ch{idx:02d}"
        self.text = (ROOT / path).read_text(encoding="utf-8")
        self.title = ""
        self.anchors: dict[str, str] = {}   # GitHub slug -> book id
        self.mermaid = 0
        self.images: list[str] = []
        self.scan()

    def scan(self) -> None:
        seen: dict[str, int] = {}
        first_h1 = True
        for is_code, chunk in split_fences(self.text):
            if is_code:
                if chunk.lstrip().startswith("```mermaid"):
                    self.mermaid += 1
                continue
            for line in chunk.split("\n"):
                m = HEADING_RE.match(line)
                if not m:
                    continue
                level, text = len(m.group(1)), m.group(2)
                text = re.sub(r"\s*\{#[^}]*\}\s*$", "", text)
                slug = gh_slug(text)
                n = seen.get(slug, 0)
                seen[slug] = n + 1
                key = slug if n == 0 else f"{slug}-{n}"
                if level == 1 and first_h1:
                    self.title = text
                    self.anchors[key] = self.cid
                    first_h1 = False
                else:
                    self.anchors[key] = f"{self.cid}-{key}"


# ------------------------------------------------------------------ figures

LIGHT_PALETTE = [   # the house style's dark card -> a light card for e-ink (FIGURE-STYLE.md section 4 and svgchart.py)
    ("#161E1C", "#FFFFFF"), ("#161e1c", "#FFFFFF"),
    ("#E4EAE7", "#111111"), ("#e4eae7", "#111111"),
    ("#97A59F", "#555555"), ("#97a59f", "#555555"),
    ("#16302A", "#E6F4F1"), ("#16302a", "#E6F4F1"),
    ("#4CC5B7", "#1E8C7F"), ("#4cc5b7", "#1E8C7F"),
    ("#7ADACF", "#136B61"), ("#7adacf", "#136B61"),
    ("#3A2C25", "#FBEADF"), ("#3a2c25", "#FBEADF"),
    ("#E39A62", "#B8601F"), ("#e39a62", "#B8601F"),
    ("#27322E", "#DDDDDD"), ("#27322e", "#DDDDDD"),
    ("#55635E", "#777777"), ("#55635e", "#777777"),
    ("#B7A3F0", "#6B4FD8"), ("#b7a3f0", "#6B4FD8"),
    ("#E6C85C", "#9A7A00"), ("#e6c85c", "#9A7A00"),
    ("#2A3531", "#DDDDDD"), ("#2a3531", "#DDDDDD"),
    ("#1F2927", "#F4F6F5"), ("#1f2927", "#F4F6F5"),
    ("#22302C", "#F0F4F2"), ("#22302c", "#F0F4F2"),
    ("#263330", "#E6EAE8"), ("#263330".lower(), "#E6EAE8"),
    ("#C9D2CE", "#333333"), ("#c9d2ce", "#333333"),
]


def lighten_svg(text: str) -> str:
    for dark, light in LIGHT_PALETTE:
        text = text.replace(dark, light)
    # a white card needs an edge so that the figure keeps its bounds on a white page
    text = re.sub(r'(<rect x="0" y="0" width="\d+" height="\d+" rx="8" fill="#FFFFFF")/>', r'\1 stroke="#CCCCCC" stroke-width="1"/>', text, count=1)
    return text


def downscale_png(path: pathlib.Path, max_w: int) -> None:
    try:
        from PIL import Image
    except ImportError:
        return
    with Image.open(path) as im:
        if im.width > max_w:
            h = round(im.height * max_w / im.width)
            im = im.resize((max_w, h), Image.LANCZOS)
            im.save(path, optimize=True)
        else:
            im.save(path, optimize=True)


# ------------------------------------------------------------------ formulas (the Kindle edition)

GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ϵ", "varepsilon": "ε", "zeta": "ζ", "eta": "η",
         "theta": "θ", "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ",
         "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "ϕ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
         "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ",
         "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω", "ell": "ℓ"}
SYMBOLS = {"cdot": "·", "times": "×", "div": "÷", "pm": "±", "mp": "∓", "approx": "≈", "le": "≤", "leq": "≤", "ge": "≥",
           "geq": "≥", "ne": "≠", "neq": "≠", "to": "→", "rightarrow": "→", "leftarrow": "←", "Rightarrow": "⇒",
           "leftrightarrow": "↔", "infty": "∞", "sim": "∼", "simeq": "≃", "in": "∈", "notin": "∉", "mid": "&#124;", "lvert": "&#124;",
           "rvert": "&#124;", "vert": "&#124;", "lVert": "‖", "rVert": "‖", "Vert": "‖", "|": "‖", "lceil": "⌈", "rceil": "⌉",
           "lfloor": "⌊", "rfloor": "⌋", "ldots": "…", "dots": "…", "cdots": "⋯", "prime": "′", "%": "%", "$": "$",
           "&": "&amp;", "#": "#", "_": "_", "{": "{", "}": "}", "top": "⊤", "partial": "∂", "nabla": "∇", "ast": "∗",
           "star": "⋆", "circ": "∘", "propto": "∝", "ll": "≪", "gg": "≫", "equiv": "≡", "subset": "⊂", "subseteq": "⊆",
           "cup": "∪", "cap": "∩", "emptyset": "∅", "forall": "∀", "exists": "∃", "neg": "¬", "land": "∧", "lor": "∨",
           "oplus": "⊕", "otimes": "⊗", "langle": "⟨", "rangle": "⟩", "degree": "°", "angle": "∠", "bullet": "•",
           "setminus": "∖", "colon": ":", "cdotp": "·", "wedge": "∧", "vee": "∨", "ge ": "≥", "hbar": "ℏ"}
SPACES = {",": "\u2009", ";": "\u2009", ":": "\u2009", "!": "", "quad": "\u2003", "qquad": "\u2003\u2003", " ": " "}
FUNCS = {"log", "ln", "exp", "min", "max", "sin", "cos", "tan", "arg", "lim", "sup", "inf", "det", "dim", "gcd", "softmax",
         "argmax", "argmin", "mod", "sgn", "KL", "var", "cov", "Pr"}
BLOBS = {"mathbb": {"R": "ℝ", "E": "𝔼", "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "C": "ℂ", "P": "ℙ", "I": "𝕀", "1": "𝟙"}}
DROP = {"left", "right", "big", "Big", "bigl", "bigr", "Bigl", "Bigr", "bigg", "Bigg", "biggl", "biggr", "displaystyle",
        "textstyle", "nolimits", "limits", "mathstrut", "allowbreak"}
IMAGE_ONLY = {"frac", "dfrac", "tfrac", "sqrt", "sum", "prod", "int", "begin", "end", "overline", "underline", "hat", "bar",
              "tilde", "vec", "dot", "ddot", "underbrace", "overbrace", "binom", "stackrel", "substack", "cfrac", "iint",
              "oint", "bigcup", "bigcap", "coprod", "widehat", "widetilde", "overset", "underset", "xrightarrow", "mathop"}


class NotSimple(Exception):
    pass


def tex_tokens(tex: str) -> list[str]:
    toks: list[str] = []
    i = 0
    while i < len(tex):
        c = tex[i]
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+|.)", tex[i:])
            toks.append(m.group(0))
            i += len(m.group(0))
        elif c in "{}^_":
            toks.append(c)
            i += 1
        elif c.isspace():
            i += 1
        else:
            toks.append(c)
            i += 1
    return toks


def tex_to_html(tex: str) -> str:
    """A small inline formula as HTML text (italic variables, sub- and superscripts, Unicode symbols). Raises NotSimple
    for anything the transliteration cannot carry faithfully (fractions, roots, sums, accents, environments, unknown
    commands); the caller renders an image then."""
    toks = tex_tokens(tex)
    pos = 0

    def group() -> str:
        nonlocal pos
        if pos < len(toks) and toks[pos] == "{":
            pos += 1
            out = []
            depth = 0
            while pos < len(toks):
                t = toks[pos]
                if t == "{":
                    depth += 1
                if t == "}":
                    if depth == 0:
                        pos += 1
                        return "".join(out)
                    depth -= 1
                out.append(atom())
            raise NotSimple("unbalanced")
        return atom()

    def text_group() -> str:
        """The raw text inside braces (for \\text, \\mathrm and friends): letters, digits, spaces and plain symbols."""
        nonlocal pos
        if pos >= len(toks) or toks[pos] != "{":
            return atom()
        pos += 1
        out = []
        while pos < len(toks) and toks[pos] != "}":
            t = toks[pos]
            if t.startswith("\\"):
                name = t[1:]
                if name in SPACES:
                    out.append(" ")
                elif name in SYMBOLS:
                    out.append(SYMBOLS[name])
                elif name in ("_", "{", "}", "%", "$", "&", "#"):
                    out.append(html.escape(name))
                else:
                    raise NotSimple(t)
            elif t in "^_":
                raise NotSimple("script in text")
            else:
                out.append(html.escape(t))
            pos += 1
        if pos >= len(toks):
            raise NotSimple("unbalanced text")
        pos += 1
        return "".join(out)

    def atom() -> str:
        nonlocal pos
        t = toks[pos]
        pos += 1
        if t == "{":
            pos -= 1
            return group()
        if t == "}":
            raise NotSimple("stray brace")
        if t in "^_":
            inner = group()
            tag = "sup" if t == "^" else "sub"
            return f"<{tag}>{inner}</{tag}>"
        if t.startswith("\\"):
            name = t[1:]
            if name in DROP:
                return ""
            if name in SPACES:
                return SPACES[name]
            if name in IMAGE_ONLY:
                raise NotSimple(name)
            if name in GREEK:
                return f"<i>{GREEK[name]}</i>"
            if name in ("mid", "vert", "lvert", "rvert"):
                return " &#124; " if name == "mid" else "&#124;"
            if name in SYMBOLS:
                return f" {SYMBOLS[name]} " if name in ("cdot", "times", "approx", "le", "leq", "ge", "geq", "ne", "neq", "to", "rightarrow", "leftarrow", "Rightarrow", "sim", "in", "propto", "ll", "gg", "equiv", "pm", "div") else SYMBOLS[name]
            if name in FUNCS:
                return f" {name} "
            if name in ("text", "mathrm", "operatorname", "textrm", "mathsf", "textsf"):
                return text_group()
            if name in ("texttt", "mathtt"):
                return f"<code>{text_group()}</code>"
            if name in ("mathit", "textit", "emph"):
                return f"<i>{text_group()}</i>"
            if name in ("mathbf", "boldsymbol", "textbf", "bm"):
                return f"<b>{group()}</b>"
            if name in ("mathcal", "mathfrak", "mathscr"):
                return f"<i>{text_group()}</i>"
            if name in BLOBS:
                inner = text_group()
                if inner in BLOBS[name]:
                    return BLOBS[name][inner]
                raise NotSimple(name)
            raise NotSimple(t)
        if t.isalpha():
            return f"<i>{t}</i>"
        if t in "=+<>":
            return f" {html.escape(t)} "
        if t == "-":
            return " − "
        if t == "*":
            return "∗"
        if t == "'":
            return "′"
        if t == "|":
            return "&#124;"      # never a bare pipe: inside a table cell pandoc would read it as a separator
        return html.escape(t)

    out = []
    while pos < len(toks):
        out.append(atom())
    s = "".join(out)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("</i><i>", "")
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+([),])", r"\1", s)
    s = re.sub(r"(<su[bp]>)\s+", r"\1", s)
    s = re.sub(r"\s+(</su[bp]>)", r"\1", s)
    return f'<span class="m">{s}</span>'


def drop_underbrace(tex: str) -> str:
    """\\underbrace{X}_{label} -> X: mathtext has no under- or overbrace; the label is in the prose."""
    out = tex
    for name in ("underbrace", "overbrace"):
        while True:
            i = out.find("\\" + name + "{")
            if i < 0:
                break
            j = i + len(name) + 2
            depth = 1
            k = j
            while k < len(out) and depth:
                depth += {"{": 1, "}": -1}.get(out[k], 0)
                k += 1
            inner = out[j:k - 1]
            rest = out[k:]
            m = re.match(r"\s*[_^]\s*(\{)", rest)
            if m:
                d, q = 1, m.end()
                while q < len(rest) and d:
                    d += {"{": 1, "}": -1}.get(rest[q], 0)
                    q += 1
                rest = rest[q:]
            else:
                m2 = re.match(r"\s*[_^]\s*(\\[A-Za-z]+|.)", rest)
                if m2:
                    rest = rest[m2.end():]
            out = out[:i] + "{" + inner + "}" + rest
    return out


class MathRenderer:
    """Formulas as PNG images through matplotlib's mathtext (Kindle devices render no MathML)."""

    def __init__(self, media: pathlib.Path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.media = media / "math"
        self.media.mkdir(parents=True, exist_ok=True)
        self.failures: list[str] = []
        self.count = 0

    @staticmethod
    def prepare(tex: str) -> list[str]:
        """Rewrite what mathtext rejects; an aligned environment becomes one line per row."""
        t = tex.strip()
        t = re.sub(r"\\(?:big|Big|bigg|Bigg)[lrm]?\b", "", t)
        t = t.replace(r"\lVert", r"\Vert").replace(r"\rVert", r"\Vert").replace(r"\lvert", "|").replace(r"\rvert", "|")
        t = t.replace(r"\texttt", r"\mathtt").replace(r"\bmod", r"\;\mathrm{mod}\;").replace(r"\operatorname*", r"\operatorname")
        t = t.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac").replace(r"\boldsymbol", r"\mathbf").replace(r"\textbf", r"\mathbf")
        t = t.replace(r"\,", r"\,").replace(r"\\;", r"\;")
        t = re.sub(r"\\label\{[^}]*\}", "", t)
        t = re.sub(r"\\(le|ge)\b", lambda m: "\\" + m.group(1) + "q", t)          # mathtext knows \leq and \geq only
        t = re.sub(r"\\(textstyle|displaystyle|scriptstyle|scriptscriptstyle)\b", "", t)
        t = re.sub(r"\\(mathrel|mathbin|mathord|mathop)\{([^{}]*)\}", r"\2", t)
        t = drop_underbrace(t)
        m = re.match(r"\\begin\{(aligned|align\*?|gather\*?|array)\}(\{[^}]*\})?(.*)\\end\{\1\}", t, re.S)
        space = lambda s: re.sub(r"(?<=[A-Za-z}\)])\s*\\(log|ln|exp|min|max|sin|cos|tan)\b", r"\\,\\\1", s)   # mathtext sets "p\log p" tight
        if m:
            body = m.group(3)
            rows = [re.sub(r"^\s*\[[^\]]*\]", "", r).strip() for r in re.split(r"\\\\", body)]
            return [space(r.replace("&", " ")) for r in rows if r.strip()]
        return [space(t)]

    def render(self, tex: str, display: bool) -> list[tuple[str, float]]:
        """[(media-relative path, height in em)] for the lines of the formula; [] when mathtext rejects it."""
        out = []
        for line in self.prepare(tex):
            key = hashlib.sha1((("d" if display else "i") + line).encode("utf-8")).hexdigest()[:16]
            png = self.media / f"{key}.png"
            fontsize = 12 if display else 11
            dpi = 220
            if not png.exists():
                fig = self.plt.figure(figsize=(0.1, 0.1))
                try:
                    txt = fig.text(0, 0, f"${line}$", fontsize=fontsize)
                    fig.canvas.draw()
                    bbox = txt.get_window_extent()
                    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=0.04, facecolor="white")
                except Exception as e:  # noqa: BLE001 - mathtext raises ValueError on syntax it does not know
                    self.failures.append(f"{line[:80]} :: {str(e).splitlines()[0][:80] if str(e) else type(e).__name__}")
                    self.plt.close(fig)
                    return []
                self.plt.close(fig)
                self.count += 1
            try:
                from PIL import Image
                with Image.open(png) as im:
                    h_px = im.height
            except Exception:  # noqa: BLE001
                h_px = 40
            em = h_px / (dpi * fontsize / 72.0)
            out.append((f"media/math/{png.name}", em))
        return out


# ------------------------------------------------------------------ the build


class Build:
    def __init__(self, edition: str, out_dir: pathlib.Path, check: bool, azw3: bool):
        self.edition = edition
        self.out = out_dir
        self.work = out_dir / f"build-{edition}"
        self.media = self.work / "media"
        self.check = check
        self.azw3 = azw3
        self.chapters: list[Chapter] = []
        self.by_path: dict[str, Chapter] = {}
        self.svg_jobs: list[tuple[pathlib.Path, pathlib.Path]] = []
        self.math: MathRenderer | None = None
        self.stats = {"links_internal": 0, "links_dropped": 0, "links_web": 0, "images": 0, "mermaid": 0,
                      "math_text": 0, "math_image": 0, "math_fallback": 0}

    # -- chapters
    def collect(self) -> None:
        idx = 0
        for part_no, (_, _, paths) in enumerate(BOOK, start=1):
            for p in paths:
                idx += 1
                ch = Chapter(idx, p, part_no)
                self.chapters.append(ch)
                self.by_path[p] = ch

    def resolve_link(self, ch: Chapter, url: str) -> str | None:
        """The book-internal target of a link, or None when the link has no place in the book."""
        if url.startswith("#"):
            frag = url[1:]
            return "#" + ch.anchors.get(frag, ch.cid)
        path, _, frag = url.partition("#")
        target = os.path.normpath(os.path.join(os.path.dirname(ch.path), path)).replace(os.sep, "/")
        other = self.by_path.get(target)
        if other is None:
            return None
        if frag and frag in other.anchors:
            return "#" + other.anchors[frag]
        return "#" + other.cid

    def process_prose(self, ch: Chapter, chunk: str, slug_seen: dict[str, int], state: dict) -> str:
        lines = chunk.split("\n")
        out: list[str] = []
        for line in lines:
            m = HEADING_RE.match(line)
            if m:
                level, text = len(m.group(1)), re.sub(r"\s*\{#[^}]*\}\s*$", "", m.group(2))
                slug = gh_slug(text)
                n = slug_seen.get(slug, 0)
                slug_seen[slug] = n + 1
                key = slug if n == 0 else f"{slug}-{n}"
                if level == 1 and not state["h1_done"]:
                    state["h1_done"] = True
                    out.append(f"## {text} {{#{ch.cid}}}")
                else:
                    new_level = min(level + 1, 6)
                    out.append(f"{'#' * new_level} {text} {{#{ch.anchors.get(key, ch.cid + '-' + key)}}}")
                continue
            im = IMG_RE.match(line)
            if im:
                out.append(self.process_image(ch, im.group("alt"), im.group("src")))
                continue
            out.append(line)
        text = "\n".join(out)
        # links, outside inline code
        protected, spans = protect_inline_code(text)

        def link(m: re.Match) -> str:
            url = m.group("url")
            if re.match(r"^(https?:|mailto:)", url):
                self.stats["links_web"] += 1
                return m.group(0)
            target = self.resolve_link(ch, url)
            if target:
                self.stats["links_internal"] += 1
                return f"[{m.group('text')}]({target})"
            self.stats["links_dropped"] += 1
            return m.group("text")

        protected = LINK_RE.sub(link, protected)
        # formulas: protected from the tag escaping; a bare price pair ($0.70/hr ... $) is not a formula; \tfrac and
        # \dfrac become \frac (texmath writes them as <mfrac displaystyle=...>, which the EPUB schema rejects)
        protected, maths = protect_math(protected)
        protected = escape_stray_tags(protected)
        protected = restore_math(protected, maths)
        if self.edition == "kindle":
            protected = self.process_math(protected)
        return restore_inline_code(protected, spans)

    def process_math(self, text: str) -> str:
        assert self.math is not None

        def display(m: re.Match) -> str:
            tex = re.sub(r"^\s*>\s?", "", m.group(1), flags=re.M).strip()
            rendered = self.math.render(tex, display=True)
            if not rendered:
                self.stats["math_fallback"] += 1
                return f"\n\n```\n{tex}\n```\n\n"
            self.stats["math_image"] += 1
            imgs = "".join(f'<img class="md" src="{p}" alt="formula" />' for p, _ in rendered)
            return f'\n\n<p class="math-display">{imgs}</p>\n\n'

        def inline(m: re.Match) -> str:
            tex = m.group(1)
            try:
                s = tex_to_html(tex)
                self.stats["math_text"] += 1
                return s
            except NotSimple:
                pass
            rendered = self.math.render(tex, display=False)
            if not rendered:
                self.stats["math_fallback"] += 1
                return "`" + tex.replace("|", "\\|") + "`"
            self.stats["math_image"] += 1
            p, em = rendered[0]
            em = max(0.9, min(em, 3.0))
            return f'<img class="mi" src="{p}" alt="formula" style="height:{em:.2f}em" />'

        text = DISPLAY_MATH_RE.sub(display, text)
        return INLINE_MATH_RE.sub(inline, text)

    def process_image(self, ch: Chapter, alt: str, src: str) -> str:
        src_path = (ROOT / ch.path).parent / src
        if not src_path.exists():
            src_path = ROOT / src
        if not src_path.exists():
            print(f"  missing image {src} in {ch.path}", file=sys.stderr)
            return f"*[missing figure: {alt}]*"
        self.stats["images"] += 1
        stem = f"{ch.cid}-{src_path.stem}"
        if src_path.suffix.lower() == ".svg":
            if self.edition == "apple":
                dst = self.media / f"{stem}.svg"
                shutil.copyfile(src_path, dst)
            else:
                light = self.work / "light" / f"{stem}.svg"
                light.parent.mkdir(parents=True, exist_ok=True)
                light.write_text(lighten_svg(src_path.read_text(encoding="utf-8")), encoding="utf-8")
                dst = self.media / f"{stem}.png"
                self.svg_jobs.append((light, dst))
        else:
            dst = self.media / f"{stem}{src_path.suffix.lower()}"
            shutil.copyfile(src_path, dst)
        return f"![{alt}](media/{dst.name})"

    def render_mermaid(self, ch: Chapter) -> list[pathlib.Path]:
        """Render the chapter's Mermaid blocks to PNG with the repository's renderer; [] when there are none."""
        if ch.mermaid == 0:
            return []
        outdir = self.work / "mermaid" / ch.cid
        outdir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        r = subprocess.run(["node", str(RENDER), "md", str(ROOT / ch.path), str(outdir)], capture_output=True, text=True, env=env)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:], file=sys.stderr)
            raise SystemExit(f"the Mermaid renderer failed on {ch.path}")
        base = pathlib.Path(ch.path).stem
        pngs = []
        for n in range(1, ch.mermaid + 1):
            src = outdir / f"{base}-m{n}.png"
            if not src.exists():
                raise SystemExit(f"renderer wrote no {src}")
            dst = self.media / f"{ch.cid}-m{n}.png"
            shutil.copyfile(src, dst)
            downscale_png(dst, 1600)
            pngs.append(dst)
        self.stats["mermaid"] += len(pngs)
        return pngs

    def process_chapter(self, ch: Chapter) -> pathlib.Path:
        pngs = self.render_mermaid(ch)
        segments = split_fences(ch.text)
        out: list[str] = []
        slug_seen: dict[str, int] = {}
        state = {"h1_done": False}
        mermaid_i = 0
        for i, (is_code, chunk) in enumerate(segments):
            if is_code:
                if chunk.lstrip().startswith("```mermaid"):
                    # the caption is the italic paragraph that follows the block; it becomes the alt text as well
                    alt = "Diagram"
                    if i + 1 < len(segments):
                        nxt = [l.strip() for l in segments[i + 1][1].split("\n") if l.strip()]
                        if nxt and nxt[0].startswith("*") and nxt[0].endswith("*"):
                            alt = nxt[0].strip("*").replace("[", "").replace("]", "")
                            alt = re.sub(r"`", "", alt)[:300]
                    png = pngs[mermaid_i]
                    mermaid_i += 1
                    out.append(f"![{alt}](media/{png.name})")
                else:
                    out.append(chunk)
            else:
                out.append(self.process_prose(ch, chunk, slug_seen, state))
        text = "\n".join(out)
        if not state["h1_done"]:
            text = f"## {ch.title or pathlib.Path(ch.path).stem} {{#{ch.cid}}}\n\n" + text
        # a note on the source, at the end of the chapter
        text += f"\n\n---\n\n*Source: `{ch.path}` in the repository, with its notebooks and labs: {REPO_URL}/tree/main/{os.path.dirname(ch.path)}*\n"
        dst = self.work / f"{ch.cid}.md"
        dst.write_text(text, encoding="utf-8")
        return dst

    # -- front matter, parts, cover, css
    def front_matter(self) -> pathlib.Path:
        n_ch = len(self.chapters)
        words = sum(len(c.text.split()) for c in self.chapters)
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        date = subprocess.run(["git", "log", "-1", "--format=%cs"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        repo_map = ROOT / "site" / "assets" / "repo-map.svg"
        map_line = ""
        if repo_map.exists():
            if self.edition == "apple":
                dst = self.media / "repo-map.svg"
                shutil.copyfile(repo_map, dst)
            else:
                light = self.work / "light" / "repo-map.svg"
                light.parent.mkdir(parents=True, exist_ok=True)
                light.write_text(lighten_svg(repo_map.read_text(encoding="utf-8")), encoding="utf-8")
                dst = self.media / "repo-map.png"
                self.svg_jobs.append((light, dst))
            map_line = f"\n![The repository map: the eight layers from the agents at the top to the hardware at the bottom, with the foundations beneath, and the topic folders of each layer](media/{dst.name})\n\n*The stack, bottom-up: each layer is built on the one below it. The parts of this book follow it.*\n"
        parts = "\n".join(f"- **Part {ROMAN[i]}: {name}.** {desc}" for i, (name, desc, _) in enumerate(BOOK))
        text = f"""# About this book {{#about}}

This book is the reading half of [full-stack-agentic-engineer]({REPO_URL}), a repository that mirrors the LLM
inference and serving stack from the GPUs at the bottom to the agents at the top. Every topic there comes in three
pieces: a primer, a from-scratch core with tests, and a lab with notebooks. This book holds the primers, {n_ch} of them,
about {words // 1000:,}k words, and nothing else: the notebooks, the exercises, the labs, the code and the deployment
notes stay in the repository and on its site ({SITE_URL}), where every notebook opens in Colab.
{map_line}
**How it is organised.** The parts follow the stack, bottom-up after the foundations:

{parts}

**How to read it.** Each primer stands on its own and starts with "The one-minute version". A first pass through the
"one-minute versions" of all primers takes an evening and gives the map; the chapters then reward reading in order,
because each layer uses the vocabulary of the one below it. The sections called "In a design review" and "Design
drills" are questions with model answers: the places to test what you kept.

**Conventions.** The prose of the layer primers is in ASD-STE100 (Simplified Technical English): short sentences,
one meaning per word, the active voice. A product fact carries a `(verify)` tag with its date, because the stacks
move quickly. A number in a worked example is computed by the topic's core and pinned by a test; a chart in the
figures is drawn by the core from the same functions. Links between primers work inside the book; a link to a
notebook, a lab or a source file is plain text here, and a web link needs a connection.

**Edition.** Built from the repository at commit `{commit}` ({date}), edition for {"Apple Books and EPUB readers" if self.edition == "apple" else "Kindle"}.
The text and the figures are under the MIT licence, like the repository.
"""
        dst = self.work / "00-about.md"
        dst.write_text(text, encoding="utf-8")
        return dst

    def part_page(self, i: int) -> pathlib.Path:
        name, desc, paths = BOOK[i]
        chapters = [self.by_path[p] for p in paths]
        toc = "\n".join(f"- [{c.title}](#{c.cid})" for c in chapters)
        text = f"# Part {ROMAN[i]}: {name} {{#part-{i + 1}}}\n\n{desc}\n\n{toc}\n"
        dst = self.work / f"part-{i + 1:02d}.md"
        dst.write_text(text, encoding="utf-8")
        return dst

    def cover(self) -> pathlib.Path | None:
        layers = ["07 Agents and applications", "06 Gateway", "05 Orchestrator", "04 Inference engine",
                  "03 Kubernetes and GPU scheduling", "02 CUDA, NCCL and runtime", "01 Hardware and fabric", "00 Foundations"]
        rows = "".join(
            f'<rect x="120" y="{1180 + i * 108}" width="1360" height="86" rx="10" fill="{"#16302A" if i in (0, 7) else "none"}" stroke="{"#4CC5B7" if i in (0, 7) else "#97A59F"}" stroke-width="2"/>'
            f'<text x="160" y="{1180 + i * 108 + 56}" font-size="40" fill="{"#7ADACF" if i in (0, 7) else "#E4EAE7"}">{html.escape(l)}</text>'
            for i, l in enumerate(layers))
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="2400" viewBox="0 0 1600 2400" role="img" aria-label="{html.escape(TITLE)}">
  <rect x="0" y="0" width="1600" height="2400" fill="#161E1C"/>
  <g font-family="Bricolage Grotesque, system-ui, sans-serif" fill="#E4EAE7">
    <text x="120" y="330" font-size="92" font-weight="700">The LLM Serving Stack,</text>
    <text x="120" y="450" font-size="92" font-weight="700">from GPUs to Agents</text>
    <text x="120" y="580" font-size="44" fill="#97A59F">{html.escape(SUBTITLE)}</text>
    <line x1="120" y1="680" x2="1480" y2="680" stroke="#4CC5B7" stroke-width="3"/>
    <text x="120" y="780" font-size="40" fill="#97A59F">{len(self.chapters)} primers · the stack, bottom-up</text>
    <text x="120" y="1120" font-size="34" fill="#97A59F">read from the bottom: each layer is built on the one below it</text>
    {rows}
    <text x="120" y="2220" font-size="48" fill="#E4EAE7">{html.escape(AUTHOR)}</text>
    <text x="120" y="2290" font-size="32" fill="#97A59F">{html.escape(REPO_URL)}</text>
  </g>
</svg>"""
        svg_path = self.work / "cover.svg"
        svg_path.write_text(svg, encoding="utf-8")
        png = self.work / "cover.png"
        jobs = [{"svg": str(svg_path), "png": str(png)}]
        (self.work / "cover-jobs.json").write_text(json.dumps(jobs))
        r = subprocess.run(["node", str(SVG2PNG), str(self.work / "cover-jobs.json"), "1"], capture_output=True, text=True)
        if r.returncode != 0:
            print("cover render failed:", r.stdout, r.stderr, file=sys.stderr)
            return None
        return png

    def css(self) -> pathlib.Path:
        text = """
body { font-family: Georgia, "Times New Roman", serif; line-height: 1.45; }
h1, h2, h3, h4, h5, h6 { font-family: Helvetica, Arial, sans-serif; line-height: 1.2; page-break-after: avoid; }
h1 { font-size: 1.8em; margin-top: 1.5em; }
h2 { font-size: 1.5em; margin-top: 1.4em; }
h3 { font-size: 1.25em; margin-top: 1.2em; }
h4 { font-size: 1.1em; }
h5, h6 { font-size: 1em; }
p { margin: 0 0 0.75em 0; text-align: left; }
img { max-width: 100%; height: auto; }
p img { display: block; margin: 0.8em auto; }
p img.mi { display: inline; margin: 0 0.05em; vertical-align: middle; }
p.math-display { text-align: center; margin: 0.6em 0; }
p.math-display img.md { display: block; margin: 0.25em auto; max-width: 94%; max-height: 12em; }
span.m { white-space: nowrap; }
span.m i { font-style: italic; }
pre { font-family: "Courier New", Courier, monospace; font-size: 0.78em; line-height: 1.3; white-space: pre-wrap; word-wrap: break-word; background: #f3f3f3; padding: 0.5em; margin: 0.6em 0; }
code { font-family: "Courier New", Courier, monospace; font-size: 0.85em; }
pre code { font-size: 1em; }
table { border-collapse: collapse; margin: 0.8em 0; font-size: 0.8em; width: 100%; }
th, td { border: 1px solid #999; padding: 0.25em 0.4em; vertical-align: top; text-align: left; }
th { background: #eee; }
blockquote { margin: 0.8em 0 0.8em 0.4em; padding: 0.1em 0 0.1em 0.8em; border-left: 3px solid #999; color: #222; }
hr { border: 0; border-top: 1px solid #999; margin: 1.5em 0; }
em.caption, p.caption { font-size: 0.9em; color: #333; }
a { color: #1E6E64; text-decoration: none; }
sup, sub { line-height: 0; font-size: 0.75em; }
"""
        dst = self.work / "book.css"
        dst.write_text(text, encoding="utf-8")
        return dst

    def metadata(self) -> pathlib.Path:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        date = subprocess.run(["git", "log", "-1", "--format=%cs"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        text = f"""---
title: "{TITLE}"
subtitle: "{SUBTITLE}"
author: "{AUTHOR}"
date: "{date}"
lang: en-GB
rights: "MIT licence, like the repository"
description: "The primers of the full-stack-agentic-engineer repository as one book: the LLM inference and serving stack from the GPUs to the agents. Built from commit {commit}."
publisher: "{REPO_URL}"
---
"""
        dst = self.work / "metadata.yaml"
        dst.write_text(text, encoding="utf-8")
        return dst

    # -- run
    def run(self) -> pathlib.Path:
        if self.work.exists():
            shutil.rmtree(self.work)
        self.media.mkdir(parents=True)
        self.collect()
        if self.edition == "kindle":
            self.math = MathRenderer(self.media)
        files: list[pathlib.Path] = [self.front_matter()]
        for i, (_, _, paths) in enumerate(BOOK):
            files.append(self.part_page(i))
            for p in paths:
                ch = self.by_path[p]
                print(f"  {ch.cid} {ch.title[:70]}")
                files.append(self.process_chapter(ch))
        if self.svg_jobs:
            jobs = [{"svg": str(s), "png": str(d)} for s, d in self.svg_jobs]
            jp = self.work / "svg-jobs.json"
            jp.write_text(json.dumps(jobs))
            r = subprocess.run(["node", str(SVG2PNG), str(jp), "2"], capture_output=True, text=True)
            print("  figures:", r.stdout.strip() or r.stderr.strip()[-400:])
            if r.returncode != 0:
                raise SystemExit("the SVG renderer failed")
            for _, d in self.svg_jobs:
                downscale_png(d, 1600)
        cover = self.cover()
        css = self.css()
        meta = self.metadata()
        out = self.out / (f"{slugify(TITLE)}.epub" if self.edition == "apple" else f"{slugify(TITLE)}-kindle.epub")
        fmt = "gfm+attributes+tex_math_dollars" if self.edition == "apple" else "gfm+attributes"
        cmd = ["pandoc", "--from", fmt, "--to", "epub3", "--toc", "--toc-depth=3", "--split-level=2", "--css", str(css), "--metadata-file", str(meta), "--resource-path", str(self.work),
               "--no-highlight", "-o", str(out)]
        if self.edition == "apple":
            cmd.append("--mathml")
        if cover:
            cmd += ["--epub-cover-image", str(cover)]
        cmd += [str(f) for f in files]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=self.work)
        if r.returncode != 0:
            print(r.stderr, file=sys.stderr)
            raise SystemExit("pandoc failed")
        warn = [l for l in r.stderr.splitlines() if "WARNING" in l]
        if warn:
            print(f"  pandoc warnings: {len(warn)} (first: {warn[0][:160]})")
        self.verify_epub(out)
        if self.check:
            self.epubcheck(out)
        if self.edition == "kindle" and self.azw3:
            self.make_azw3(out)
        if self.math:
            print(f"  formulas: {self.stats['math_text']} as text, {self.stats['math_image']} as images, {self.stats['math_fallback']} left as TeX; {len(self.math.failures)} mathtext failures")
            for f in self.math.failures[:12]:
                print("    ", f)
        print(f"  links: {self.stats['links_internal']} inside the book, {self.stats['links_dropped']} to notebooks or files made plain, {self.stats['links_web']} web; figures: {self.stats['images']} images, {self.stats['mermaid']} diagrams")
        return out

    def verify_epub(self, out: pathlib.Path) -> None:
        """Every internal link resolves to an id in the book, every image reference to a file in the archive."""
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(out) as z:
            names = set(z.namelist())
            ids: dict[str, set[str]] = {}
            docs = [n for n in names if n.endswith(".xhtml")]
            for n in docs:
                root = ET.fromstring(z.read(n))
                ids[n] = {el.get("id") for el in root.iter() if el.get("id")}
            all_ids = {(n, i) for n, s in ids.items() for i in s}
            bad_links = 0
            bad_imgs = 0
            for n in docs:
                root = ET.fromstring(z.read(n))
                base = os.path.dirname(n)
                for el in root.iter():
                    href = el.get("href")
                    if href and not re.match(r"^[a-z]+:", href) and "#" in href:
                        f, frag = href.split("#", 1)
                        target_doc = os.path.normpath(os.path.join(base, f)).replace(os.sep, "/") if f else n
                        if (target_doc, frag) not in all_ids:
                            bad_links += 1
                    src = el.get("src")
                    if src and not re.match(r"^[a-z]+:", src):
                        p = os.path.normpath(os.path.join(base, src)).replace(os.sep, "/")
                        if p not in names:
                            bad_imgs += 1
            size = out.stat().st_size / 1e6
            print(f"  {out.name}: {size:.1f} MB, {len(docs)} documents, {sum(1 for n in names if n.startswith('EPUB/media/'))} media files, {bad_links} unresolved links, {bad_imgs} missing images")
            if bad_links or bad_imgs:
                raise SystemExit("the EPUB has unresolved links or missing images")

    def epubcheck(self, out: pathlib.Path) -> None:
        try:
            import epubcheck  # noqa: F401
        except ImportError:
            print("  epubcheck not installed; skipped")
            return
        r = subprocess.run([sys.executable, "-m", "epubcheck", str(out)], capture_output=True, text=True)
        tail = (r.stdout + r.stderr).strip().splitlines()
        summary = [l for l in tail if "ERROR" in l or "FATAL" in l or "WARNING" in l or "Messages" in l or "valid" in l.lower()]
        print("  epubcheck:", " | ".join(summary[-4:]) if summary else (tail[-1] if tail else "no output"))
        errors = [l for l in tail if l.startswith(("ERROR", "FATAL"))]
        for l in errors[:10]:
            print("    ", l[:200])

    def make_azw3(self, epub: pathlib.Path) -> None:
        exe = shutil.which("ebook-convert")
        if not exe:
            print("  ebook-convert (Calibre) not installed; the Kindle EPUB is the deliverable (Send to Kindle accepts EPUB)")
            return
        azw3 = epub.with_suffix(".azw3")
        cmd = [exe, str(epub), str(azw3), "--output-profile", "kindle_pw3", "--no-inline-toc",
               "--chapter", "//*[name()='h1' or name()='h2']", "--level1-toc", "//h:h1", "--level2-toc", "//h:h2",
               "--level3-toc", "//h:h3", "--title", TITLE, "--authors", AUTHOR]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print("  ebook-convert failed:", r.stderr[-800:], file=sys.stderr)
            return
        print(f"  {azw3.name}: {azw3.stat().st_size / 1e6:.1f} MB")


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "build" / "book"), help="output folder (default build/book, gitignored)")
    ap.add_argument("--edition", choices=["apple", "kindle", "both"], default="both")
    ap.add_argument("--no-azw3", action="store_true", help="skip Calibre's AZW3 conversion")
    ap.add_argument("--no-check", action="store_true", help="skip epubcheck")
    args = ap.parse_args()
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    editions = ["apple", "kindle"] if args.edition == "both" else [args.edition]
    for ed in editions:
        print(f"== {ed} edition")
        Build(ed, out, check=not args.no_check, azw3=not args.no_azw3).run()


if __name__ == "__main__":
    main()
