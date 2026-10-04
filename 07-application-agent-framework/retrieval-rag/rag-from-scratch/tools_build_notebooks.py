"""Generate the teaching notebooks (with blanks) and their solutions.

Each notebook is a list of cells. Exercise cells carry BOTH a blanked version
(what the learner gets) and a solution; `write()` picks one. Self-check
`assert`s are identical in both and test the *shape* of your implementation
(sorted, correct length, right maths) rather than model-dependent rankings, so
they pass offline and under the real embedding model alike.

Every written notebook starts with the repo's Colab setup cell, exactly as
tools/inject_colab_bootstrap.py writes it, and in the injector's JSON layout, so a
rebuild reproduces the committed notebooks byte for byte and running the injector
afterwards is a no-op.

Run: `python tools_build_notebooks.py`
"""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "tools" / "inject_colab_bootstrap.py").is_file())
_spec = importlib.util.spec_from_file_location("inject_colab_bootstrap", REPO / "tools" / "inject_colab_bootstrap.py")
_inject = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_inject)
NB_DIR = HERE / "notebooks"
SOL_DIR = HERE / "solutions"
NB_DIR.mkdir(exist_ok=True)
SOL_DIR.mkdir(exist_ok=True)


# ---- tiny cell DSL --------------------------------------------------------- #
def md(text):
    return {"t": "md", "src": text.strip("\n") + "\n"}


def code(text):
    return {"t": "code", "src": text.strip("\n")}


def ex(blank, solution):
    return {"t": "code", "src": blank.strip("\n"), "sol": solution.strip("\n")}


SETUP = code("""
# --- setup: make `import ragkit` work from notebooks/ or solutions/ ---
import sys, os
sys.path.insert(0, os.path.abspath(".."))
import numpy as np
from ragkit.corpus import load_documents, load_corpus, load_qrels, tokenize
from ragkit.embed import get_embedder
from ragkit import llm
""")


# =========================================================================== #
# 00 — setup & corpus
# =========================================================================== #
NB00 = [
    md("""
# 00 · Setup & the corpus

RAG has exactly two parts. The first part **retrieves** the relevant text. Then
the second part **generates** an answer that depends on that text. This repo
teaches the retrieval half. You build that half from scratch, because almost all
RAG failures occur in it.

**What you need**

```
pip install -r requirements.txt        # T0: numpy + pytest, no torch
pip install -r requirements-full.txt   # optional: sentence-transformers (and torch) for the real models
```

If you do not have `sentence-transformers`, `ragkit.embed` uses a labelled hashing
embedder as a fallback. This embedder is lexical, not semantic. Every notebook
still runs. If you have `sentence-transformers`, the first embedding call downloads
a ~90 MB model (`all-MiniLM-L6-v2`). Then the model runs offline on the CPU.

**Generation** is optional. To use a real model, set `ANTHROPIC_API_KEY` or
`OPENAI_API_KEY`. If you do not set a key, a deterministic *extractive* fallback
answers from the retrieved text. Thus every notebook still runs.
"""),
    SETUP,
    code("""
# Which generation backend is active?
print("generation backend:", llm.available())

# Is the embedding model installed?
try:
    emb = get_embedder()
    print("embedder ready:", emb.model_name, "| dim =", emb.dim)
except ImportError as e:
    print("EMBEDDER NOT INSTALLED — run: pip install sentence-transformers\\n")
    print(e)
"""),
    md("""
## The corpus

The corpus has nine short Markdown docs for a fictional SaaS company, "Meridian".
Each doc has `##` sections (that structure is important in notebook 02). A small
labelled question set (`qrels`) lets us *measure* retrieval in notebook 05.
"""),
    code("""
docs = load_documents()
print(len(docs), "documents:\\n", ", ".join(docs))

print("\\n--- sample document: expense-policy ---\\n")
print(docs["expense-policy"])
"""),
    code("""
qrels = load_qrels()
from collections import Counter
print("questions by kind:", dict(Counter(q["kind"] for q in qrels)), "\\n")
for kind in ("lexical", "semantic", "multihop"):
    ex_q = next(q for q in qrels if q["kind"] == kind)
    print(f"[{kind:8}] {ex_q['question']}\\n           gold -> {ex_q['gold_docs']}")
"""),
    md("""
## The one idea behind dense retrieval

An **embedding model** maps text to a vector, so that *texts with similar meanings
get vectors that are near each other*. Thus "retrieval" is only two steps: embed
the query, then find the nearest document vectors. The next cell shows this
directly.
"""),
    code("""
# (needs the embedder installed)
pairs = [
    ("How many holidays do I get?", "annual leave and vacation days"),   # related
    ("How many holidays do I get?", "the API returns HTTP 429 when throttled"),  # unrelated
]
for a, b in pairs:
    va, vb = get_embedder().encode(a), get_embedder().encode(b)
    print(f"cos = {float(va @ vb):+.3f}   {a!r}  vs  {b!r}")
# Rows are L2-normalised, so the dot product IS cosine similarity.
"""),
    md("""
**Learning path**

| nb | idea |
|----|------|
| 01 | the minimal RAG loop: embed, cosine search, prompt, generate |
| 02 | chunking: why *how* you divide the text decides what you can retrieve |
| 03 | hybrid search: BM25 (exact) and dense (meaning), fused with RRF |
| 04 | reranking: low-cost recall, then a cross-encoder for precision |
| 05 | evaluation: hit@k, recall@k and MRR, to measure everything from 01 to 04 |
| 06 | iterative RAG (advanced): multi-hop questions need >1 retrieval |

Do the exercises in each notebook. Fill in the `# YOUR CODE HERE` blanks. The
`assert`s examine your work. The solutions are in `solutions/`.
"""),
]


# =========================================================================== #
# 01 — minimal RAG
# =========================================================================== #
NB01 = [
    md("""
# 01 · The minimal RAG loop

This is the full pattern, one time:

```
embed the corpus  ─┐
embed the query  ──┤→ cosine similarity → top-k chunks → prompt → generate
```

We start with the simplest chunking possible: **one chunk per document**. We do
this so that nothing takes your attention away from the loop. Notebook 02
repairs the chunking.
"""),
    SETUP,
    code("""
emb = get_embedder()
chunks = load_corpus()                 # whole documents as chunks
matrix = emb.encode([c.text for c in chunks])   # (N, dim), rows normalised
print("index:", matrix.shape, "for", len(chunks), "chunks")
"""),
    md("""
### Exercise 1 — cosine search

The rows of `matrix` are L2-normalised. Thus the cosine similarity is only the
dot product. Write the function `search`. Give each chunk a score against the
query. Return the top-`k` as `(chunk, score)` pairs, with the highest score first.
"""),
    ex(
        blank="""
def search(query, k=3):
    qv = emb.encode(query)                     # (dim,)
    # YOUR CODE HERE:
    #   1) scores = cosine of qv against every row of `matrix`  -> shape (N,)
    #   2) take the indices of the top-k scores, highest first
    #   3) return [(chunks[i], float(scores[i])), ...]
    scores = ...
    top = ...
    return ...

res = search("How many vacation days do I get each year?", k=3)
assert len(res) == 3
assert all(-1.01 <= s <= 1.01 for _, s in res)          # cosine range
assert res[0][1] >= res[1][1] >= res[2][1]              # sorted, descending
print("top-3:", [(c.doc_id, round(s, 3)) for c, s in res])
""",
        solution="""
def search(query, k=3):
    qv = emb.encode(query)                     # (dim,)
    scores = matrix @ qv                       # cosine, shape (N,)
    top = np.argsort(-scores)[:k]              # indices, best first
    return [(chunks[i], float(scores[i])) for i in top]

res = search("How many vacation days do I get each year?", k=3)
assert len(res) == 3
assert all(-1.01 <= s <= 1.01 for _, s in res)          # cosine range
assert res[0][1] >= res[1][1] >= res[2][1]              # sorted, descending
print("top-3:", [(c.doc_id, round(s, 3)) for c, s in res])
""",
    ),
    md("""
### Assemble a grounded prompt

The retrieval is complete. Now put the evidence in front of the model. Use a
numbered source list and an instruction to answer only from that evidence. The
source list and the instruction make the answer *grounded* and *citable*. This
part is infrastructure code, so we wrote it for you. Read it.
"""),
    code("""
def build_prompt(query, retrieved):
    blocks = []
    for i, (c, _) in enumerate(retrieved, 1):
        blocks.append(f"[{i}] (source: {c.doc_id})\\n{c.text}")
    context = "\\n\\n".join(blocks)
    system = ("Answer the question using ONLY the sources below. "
              "Cite the source number in square brackets after each claim. "
              "If the sources don't contain the answer, say you don't know.")
    prompt = f"{system}\\n\\n=== SOURCES ===\\n{context}\\n\\n=== QUESTION ===\\n{query}"
    return prompt, [c.text for c, _ in retrieved]

p, ctxs = build_prompt("How many vacation days do I get each year?", res)
print(p[:600], "...")
"""),
    md("""
### Exercise 2 — the end-to-end `answer()`

Connect the parts: retrieve, then build the prompt, then generate. Use
`llm.complete(...)`. Pass `contexts=` and `query=`, so that the offline fallback
has something to extract. Also pass `embedder=`, so that the fallback can rank
sentences by meaning.
"""),
    ex(
        blank="""
def answer(query, k=3):
    retrieved = search(query, k)
    prompt, ctxs = build_prompt(query, retrieved)
    # YOUR CODE HERE: return llm.complete(...) grounded in `ctxs`.
    # Pass: prompt, contexts=ctxs, query=query, embedder=emb
    return ...

out = answer("How much can I get reimbursed for food abroad per day?")
assert isinstance(out, str) and len(out) > 0
print(out)
""",
        solution="""
def answer(query, k=3):
    retrieved = search(query, k)
    prompt, ctxs = build_prompt(query, retrieved)
    return llm.complete(prompt, contexts=ctxs, query=query, embedder=emb)

out = answer("How much can I get reimbursed for food abroad per day?")
assert isinstance(out, str) and len(out) > 0
print(out)
""",
    ),
    md("""
That is a RAG system that works. Everything after this notebook makes the
**retrieve** step better. The reason is this: if the correct text never gets into
`ctxs`, no quantity of prompt work and no model quality can save the answer.
"""),
]


# =========================================================================== #
# 02 — chunking
# =========================================================================== #
NB02 = [
    md("""
# 02 · Chunking

Whole-document chunks (nb 01) waste the context window. They also mix many topics
into one vector, and that vector becomes unclear. But a simple split has the
opposite failure. It cuts sentences in half, and it removes the heading that gave
them meaning.

The key step is this: **keep what you match on separate from what you show the model**.
"""),
    SETUP,
    code("""
docs = load_documents()
sample = docs["security-access"]
print(sample)
"""),
    md("""
### Baseline — fixed-size windows

Divide the text into fixed-size word windows with a small overlap. This method is
simple, and it does not see the structure.
"""),
    code("""
def fixed_size_chunks(doc_id, text, size=40, overlap=10):
    words = text.split()
    step = max(1, size - overlap)
    out = []
    for i in range(0, len(words), step):
        window = words[i:i + size]
        if not window:
            break
        out.append(" ".join(window))
        if i + size >= len(words):
            break
    return out

fx = fixed_size_chunks("security-access", sample)
print(f"{len(fx)} chunks. Notice a chunk can start mid-sentence:\\n")
print("-", fx[1][:160], "...")
"""),
    md("""
### Exercise 1 — structure-aware chunks

Markdown already shows the boundaries. Divide the text at the `##` headings. **Put
the doc title and the heading at the start of each chunk**, so that each chunk
describes itself.

Compare two chunks. One chunk says *"Access Control Standard
(SEC-011) > Production Access"*. The other chunk is a paragraph with no header.
This paragraph starts with "Access to production...". Retrieval finds the first
chunk far better.

Return a list of `(section_heading, chunk_text)`. Each `chunk_text` must start
with the header line `"<title> > <heading>"`.
"""),
    ex(
        blank="""
def structure_aware_chunks(doc_id, text):
    lines = text.splitlines()
    title = lines[0].lstrip("# ").strip()      # first line is "# Title"
    out, heading, buf = [], None, []

    def flush():
        # YOUR CODE HERE: if we have a heading and some body text, append
        #   (heading, f"{title} > {heading}\\n{body}") to `out`.
        ...

    for line in lines:
        if line.startswith("## "):
            flush()
            heading, buf = line[3:].strip(), []
        elif not line.startswith("# "):
            buf.append(line)
    flush()
    return out

sec = structure_aware_chunks("security-access", sample)
headings = [h for h, _ in sec]
assert "Production Access" in headings
prod = next(t for h, t in sec if h == "Production Access")
assert prod.startswith("Access Control Standard")        # doc title kept as header
assert "> Production Access" in prod                      # heading in the header line
assert "VPN" in prod                                                     # body preserved
print("sections:", headings)
print("\\n", prod)
""",
        solution="""
def structure_aware_chunks(doc_id, text):
    lines = text.splitlines()
    title = lines[0].lstrip("# ").strip()      # first line is "# Title"
    out, heading, buf = [], None, []

    def flush():
        if heading and buf:
            body = "\\n".join(buf).strip()
            if body:
                out.append((heading, f"{title} > {heading}\\n{body}"))

    for line in lines:
        if line.startswith("## "):
            flush()
            heading, buf = line[3:].strip(), []
        elif not line.startswith("# "):
            buf.append(line)
    flush()
    return out

sec = structure_aware_chunks("security-access", sample)
headings = [h for h, _ in sec]
assert "Production Access" in headings
prod = next(t for h, t in sec if h == "Production Access")
assert prod.startswith("Access Control Standard")        # doc title kept as header
assert "> Production Access" in prod                      # heading in the header line
assert "VPN" in prod                                                     # body preserved
print("sections:", headings)
print("\\n", prod)
""",
    ),
    md("""
### Small-to-big (sentence-window)

This method gives precision *and* context. Match on a small unit (a sentence), but
**return** the parent section of that unit. You search at a fine granularity.
Then you give the model sufficient text around the match to answer. The reference
`sentence_chunks` does this. We import it, and we do not build it again.
"""),
    code("""
from ragkit.reference import sentence_chunks
small, parent_of = sentence_chunks("security-access", sample)
print(f"{len(small)} sentence-level match units -> {len(set(p.chunk_id for p in parent_of.values()))} parent sections\\n")
s = small[3]
print("match unit :", s.text)
print("return this:", parent_of[s.chunk_id].text[:120], "...")
"""),
    md("""
**Takeaway.** Structure-aware chunks with prepended headers are the highest-ROI
default for document corpora. Small-to-big adds precision when sections are
long. We will measure the difference with numbers in notebook 05.
"""),
]


# =========================================================================== #
# 03 — hybrid search
# =========================================================================== #
NB03 = [
    md("""
# 03 · Hybrid search — BM25 + dense, fused with RRF

Dense retrieval matches *meaning*. But it can make errors on exact tokens, for
example error codes, policy IDs and product names. Lexical **BM25** gets those
tokens correct, but it misses paraphrases. Systems in production run both and
fuse the results. We build BM25 from scratch. Then we fuse the results with
**Reciprocal Rank Fusion**.
"""),
    SETUP,
    code("""
from ragkit.reference import structure_aware_chunks   # from nb 02
from ragkit.corpus import Chunk

docs = load_documents()
chunks = []
for doc_id, text in docs.items():
    for i, (heading, ctext) in enumerate(
        [(c.section, c.text) for c in structure_aware_chunks(doc_id, text)]
    ):
        chunks.append(Chunk(f"{doc_id}#{i}", doc_id, ctext, heading))
print(len(chunks), "chunks indexed")
"""),
    md(r"""
### Exercise 1 — BM25 scoring

BM25 gives a query term $t$ in document $i$ this score:

$$
\mathrm{idf}(t) \cdot \frac{f(t,i) \cdot (k_1 + 1)}{f(t,i) + k_1 \cdot (1 - b + b \cdot \lvert d_i \rvert / \mathrm{avgdl})}
$$

In this formula, ${f(t,i)}$ is the frequency of the term in doc $i$. The length of
doc $i$ is $\lvert d_i \rvert$, and $\mathrm{avgdl}$ is the mean length. Fill in
that formula. The code already calculates the `idf` and the counts.
"""),
    ex(
        blank="""
import math
from collections import Counter

class BM25:
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = corpus_tokens
        self.N = len(corpus_tokens)
        self.avgdl = sum(len(d) for d in corpus_tokens) / self.N
        self.tf = [Counter(d) for d in corpus_tokens]
        df = Counter()
        for d in corpus_tokens:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def score(self, q_tokens, i):
        dl = len(self.docs[i])
        s = 0.0
        for t in q_tokens:
            f = self.tf[i].get(t, 0)
            if f == 0:
                continue
            # YOUR CODE HERE: add this term's BM25 contribution to `s`
            #   numerator   = f * (self.k1 + 1)
            #   denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            #   s += self.idf.get(t, 0.0) * numerator / denominator
            ...
        return s

    def search(self, query, k=5):
        q = tokenize(query)
        ranked = sorted(((i, self.score(q, i)) for i in range(self.N)),
                        key=lambda x: x[1], reverse=True)
        return ranked[:k]

bm = BM25([tokenize(c.text) for c in chunks])
# hand-checkable: more occurrences of a term -> higher score, absent term -> 0
toy = BM25([["cat", "cat", "mat"], ["cat", "hat"], ["dog"]])
assert toy.score(["cat"], 0) > toy.score(["cat"], 1) > 0
assert toy.score(["cat"], 2) == 0.0
# on the real corpus, an exact code lands the right doc
top_i, _ = bm.search("what does ERR_4290 mean", k=1)[0]
print("ERR_4290 ->", chunks[top_i].doc_id)
""",
        solution="""
import math
from collections import Counter

class BM25:
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = corpus_tokens
        self.N = len(corpus_tokens)
        self.avgdl = sum(len(d) for d in corpus_tokens) / self.N
        self.tf = [Counter(d) for d in corpus_tokens]
        df = Counter()
        for d in corpus_tokens:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def score(self, q_tokens, i):
        dl = len(self.docs[i])
        s = 0.0
        for t in q_tokens:
            f = self.tf[i].get(t, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += self.idf.get(t, 0.0) * numerator / denominator
        return s

    def search(self, query, k=5):
        q = tokenize(query)
        ranked = sorted(((i, self.score(q, i)) for i in range(self.N)),
                        key=lambda x: x[1], reverse=True)
        return ranked[:k]

bm = BM25([tokenize(c.text) for c in chunks])
# hand-checkable: more occurrences of a term -> higher score, absent term -> 0
toy = BM25([["cat", "cat", "mat"], ["cat", "hat"], ["dog"]])
assert toy.score(["cat"], 0) > toy.score(["cat"], 1) > 0
assert toy.score(["cat"], 2) == 0.0
# on the real corpus, an exact code lands the right doc
top_i, _ = bm.search("what does ERR_4290 mean", k=1)[0]
print("ERR_4290 ->", chunks[top_i].doc_id)
""",
    ),
    md("""
### See where each method wins

Build a dense index over the same chunks. Then compare the two retrievers on a
**lexical** query (an exact code) and a **semantic** query (a paraphrase).
"""),
    code("""
emb = get_embedder()
mat = emb.encode([c.text for c in chunks])

def dense_ids(query, k=5):
    qv = emb.encode(query)
    order = np.argsort(-(mat @ qv))[:k]
    return [chunks[i].chunk_id for i in order]

def bm25_ids(query, k=5):
    return [chunks[i].chunk_id for i, _ in bm.search(query, k)]

for q in ["what does ERR_4290 mean",                       # lexical
          "am I allowed to work from home every day"]:      # semantic
    print(f"\\nQ: {q}")
    print("  dense:", [i.split('#')[0] for i in dense_ids(q, 3)])
    print("  bm25 :", [i.split('#')[0] for i in bm25_ids(q, 3)])
"""),
    md(r"""
### Exercise 2 — Reciprocal Rank Fusion

RRF combines ranked lists with only the **rank position**. Thus the different
scales of BM25 scores and cosine scores have no effect on RRF:

$$
\mathrm{score}(d) = \sum_{\text{lists}} \frac{1}{k + r(d)}
$$

Here $k = 60$, and $r(d) = 1$ for the top result of a list.

`enumerate` counts from 0. Thus in code, the term is `1 / (k + rank + 1)`.

Write RRF. Then fuse the two ranked lists from dense search and BM25.
"""),
    ex(
        blank="""
from collections import defaultdict

def rrf(rankings, k=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            # YOUR CODE HERE: add 1 / (k + rank + 1) to scores[doc_id]
            ...
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

# consensus check: 'b' is near the top of both lists, so it should win
fused = rrf([["a", "b", "c"], ["b", "c", "a"]])
assert fused == ["b", "a", "c"], fused   # 1/61+1/62 > 1/61+1/63 > 1/62+1/63

# exact-formula check. With k=60 an off-by-one in the rank barely moves a score, so
# the probe needs deep ranks. Four documents, filler everywhere else:
#   x1: 1st in A                  -> 1/61          = 0.016393
#   y1: 38th in B and 100th in A  -> 1/98 + 1/160  = 0.016454
#   x2: 2nd in B                  -> 1/62          = 0.016129
#   y2: 40th in A and 98th in B   -> 1/100 + 1/158 = 0.016329
# Ranks counted from 0 put x1 above y1; ranks from 2 put y2 above x1.
def ranked(placed, n, filler):
    return [placed.get(r, f"{filler}{r}") for r in range(1, n + 1)]
A = ranked({1: "x1", 40: "y2", 100: "y1"}, 100, "a")
B = ranked({2: "x2", 38: "y1", 98: "y2"}, 100, "b")
probe = [d for d in rrf([A, B]) if d in {"x1", "y1", "x2", "y2"}]
assert probe == ["y1", "x1", "y2", "x2"], f"{probe}: use 1 / (k + rank + 1) with k=60"

def hybrid_ids(query, k=5):
    return rrf([dense_ids(query, 10), bm25_ids(query, 10)])[:k]

for q in ["what does ERR_4290 mean", "am I allowed to work from home every day"]:
    print(q, "->", [i.split('#')[0] for i in hybrid_ids(q, 3)])
""",
        solution="""
from collections import defaultdict

def rrf(rankings, k=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

# consensus check: 'b' is near the top of both lists, so it should win
fused = rrf([["a", "b", "c"], ["b", "c", "a"]])
assert fused == ["b", "a", "c"], fused   # 1/61+1/62 > 1/61+1/63 > 1/62+1/63

# exact-formula check. With k=60 an off-by-one in the rank barely moves a score, so
# the probe needs deep ranks. Four documents, filler everywhere else:
#   x1: 1st in A                  -> 1/61          = 0.016393
#   y1: 38th in B and 100th in A  -> 1/98 + 1/160  = 0.016454
#   x2: 2nd in B                  -> 1/62          = 0.016129
#   y2: 40th in A and 98th in B   -> 1/100 + 1/158 = 0.016329
# Ranks counted from 0 put x1 above y1; ranks from 2 put y2 above x1.
def ranked(placed, n, filler):
    return [placed.get(r, f"{filler}{r}") for r in range(1, n + 1)]
A = ranked({1: "x1", 40: "y2", 100: "y1"}, 100, "a")
B = ranked({2: "x2", 38: "y1", 98: "y2"}, 100, "b")
probe = [d for d in rrf([A, B]) if d in {"x1", "y1", "x2", "y2"}]
assert probe == ["y1", "x1", "y2", "x2"], f"{probe}: use 1 / (k + rank + 1) with k=60"

def hybrid_ids(query, k=5):
    return rrf([dense_ids(query, 10), bm25_ids(query, 10)])[:k]

for q in ["what does ERR_4290 mean", "am I allowed to work from home every day"]:
    print(q, "->", [i.split('#')[0] for i in hybrid_ids(q, 3)])
""",
    ),
    md("""
Hybrid gets the exact-code query *and* the paraphrase correct. In notebook 05, we
will show with numbers that hybrid is better than each method alone, across the
whole query set.
"""),
]


# =========================================================================== #
# 04 — reranking
# =========================================================================== #
NB04 = [
    md("""
# 04 · Reranking

First-stage retrievers (dense, BM25) are *low-cost* by design, and they give
priority to recall. Their job is to get the correct chunk somewhere in the top
20–100. Then a **cross-encoder** reads each `(query, chunk)` pair *together*. It
gives a relevance score that is far more accurate. It is too slow to run over the
whole corpus, but it is perfect over a shortlist.

```
retrieve top-N (cheap)  →  cross-encoder rerank  →  keep top-k (precise)
```
"""),
    SETUP,
    code("""
from ragkit.reference import structure_aware_chunks, chunk_corpus
docs = load_documents()
chunks = chunk_corpus(docs, structure_aware_chunks)
emb = get_embedder()
mat = emb.encode([c.text for c in chunks])

def retrieve(query, n=10):
    qv = emb.encode(query)
    order = np.argsort(-(mat @ qv))[:n]
    return [chunks[i] for i in order]
"""),
    md("""
### Exercise — two-stage retrieve-then-rerank

`get_cross_encoder()` returns a model with `.predict([(query, passage), ...])`.
This method gives a relevance score for each pair. Retrieve `n` candidates. Give
each candidate a score. Return the top `k` chunks by cross-encoder score.
"""),
    ex(
        blank="""
from ragkit.embed import get_cross_encoder
reranker = get_cross_encoder()

def rerank(query, n=10, k=3):
    candidates = retrieve(query, n)
    # YOUR CODE HERE:
    #   pairs  = [(query, c.text) for c in candidates]
    #   scores = reranker.predict(pairs)
    #   return the k candidates with the highest scores (highest first)
    ...

q = "how quickly must someone be paged for a customer-facing outage"
top = rerank(q, n=10, k=3)
assert len(top) == 3
assert all(hasattr(c, "chunk_id") for c in top)
print("reranked top-3:", [c.chunk_id for c in top])
""",
        solution="""
from ragkit.embed import get_cross_encoder
reranker = get_cross_encoder()

def rerank(query, n=10, k=3):
    candidates = retrieve(query, n)
    pairs = [(query, c.text) for c in candidates]
    scores = reranker.predict(pairs)
    order = np.argsort(-np.asarray(scores))[:k]
    return [candidates[i] for i in order]

q = "how quickly must someone be paged for a customer-facing outage"
top = rerank(q, n=10, k=3)
assert len(top) == 3
assert all(hasattr(c, "chunk_id") for c in top)
print("reranked top-3:", [c.chunk_id for c in top])
""",
    ),
    md("""
### Watch it fix an ordering

Compare the rank of the truly-relevant chunk before and after reranking. The
cross-encoder usually moves the exact-answer chunk up. First-stage cosine search
put that chunk below its topically-similar neighbours.
"""),
    code("""
def dense_order(query, n=10):
    return retrieve(query, n)

q = "how quickly must someone be paged for a customer-facing outage"
before = [c.chunk_id for c in dense_order(q, 10)]
after = [c.chunk_id for c in rerank(q, n=10, k=10)]
gold_doc = "incident-response"
def first_hit(order):
    return next((r for r, cid in enumerate(order, 1) if cid.split('#')[0] == gold_doc), None)
print("rank of incident-response chunk  before:", first_hit(before), " after:", first_hit(after))
"""),
    md("""
Reranking is usually the single best precision improvement after hybrid retrieval.
It is also a drop-in step: retrieve many candidates, rerank them, keep a few.
"""),
]


# =========================================================================== #
# 05 — evaluation
# =========================================================================== #
NB05 = [
    md("""
# 05 · Evaluation

You cannot improve what you do not measure. A statement such as "the demo looked
good" is not a measurement. With the labelled `qrels`, we can give each retriever
a score with two standard metrics:

* **Hit@k**: is *any* gold document in the top $k$? (Did we get something useful?)
* **Recall@k**: are *all* the gold documents in the top $k$? (Did we get everything
  that the answer needs?) On a one-gold question, the two metrics agree. On a
  two-gold (multihop) question, hit@k can show 1.0 while the results do not have
  half of the answer.
* **MRR**: 1/rank of the first gold hit. (How high was its rank?)
"""),
    SETUP,
    md("""
### Exercise 1 — the metrics

Write both metrics over **document-level** ranked lists (the gold labels are per
document). `recall_at_k(..., require_all=False)` is hit@k. With
`require_all=True`, it is recall@k. These are pure functions. The asserts compare
the results against the exact maths.
"""),
    ex(
        blank="""
def recall_at_k(ranked_docs, gold_docs, k, require_all=False):
    top = ranked_docs[:k]
    # YOUR CODE HERE: 1.0 if ANY gold doc is in `top` (ALL of them when require_all), else 0.0
    return ...

def mrr(ranked_docs, gold_docs):
    # YOUR CODE HERE: 1/rank of the first gold doc (rank starts at 1), else 0.0
    ...

assert recall_at_k(["a", "b", "c"], ["c"], k=3) == 1.0
assert recall_at_k(["a", "b", "c"], ["c"], k=2) == 0.0
# two gold documents: hit@k needs one of them, recall@k needs both
assert recall_at_k(["a", "b", "c"], ["a", "z"], k=3) == 1.0
assert recall_at_k(["a", "b", "c"], ["a", "z"], k=3, require_all=True) == 0.0
assert recall_at_k(["a", "b", "c"], ["a", "c"], k=3, require_all=True) == 1.0
assert recall_at_k(["a", "b", "c"], ["a", "c"], k=2, require_all=True) == 0.0
assert mrr(["a", "b", "c"], ["b"]) == 0.5
assert mrr(["a", "b", "c"], ["z"]) == 0.0
print("metrics OK")
""",
        solution="""
def recall_at_k(ranked_docs, gold_docs, k, require_all=False):
    top = ranked_docs[:k]
    hits = [g in top for g in gold_docs]
    return 1.0 if (all(hits) if require_all else any(hits)) else 0.0

def mrr(ranked_docs, gold_docs):
    for rank, d in enumerate(ranked_docs, start=1):
        if d in gold_docs:
            return 1.0 / rank
    return 0.0

assert recall_at_k(["a", "b", "c"], ["c"], k=3) == 1.0
assert recall_at_k(["a", "b", "c"], ["c"], k=2) == 0.0
# two gold documents: hit@k needs one of them, recall@k needs both
assert recall_at_k(["a", "b", "c"], ["a", "z"], k=3) == 1.0
assert recall_at_k(["a", "b", "c"], ["a", "z"], k=3, require_all=True) == 0.0
assert recall_at_k(["a", "b", "c"], ["a", "c"], k=3, require_all=True) == 1.0
assert recall_at_k(["a", "b", "c"], ["a", "c"], k=2, require_all=True) == 0.0
assert mrr(["a", "b", "c"], ["b"]) == 0.5
assert mrr(["a", "b", "c"], ["z"]) == 0.0
print("metrics OK")
""",
    ),
    md("""
### Exercise 2 — the evaluation harness

`evaluate` takes a retrieval function `run(query) -> ranked list of doc_ids`. It
calculates the mean of hit@k, recall@k (every gold document) and MRR over a set of
questions. Fill in the loop that calculates the means. The assert uses a fake
`run` with a known answer, so the assert does not depend on the model.
"""),
    ex(
        blank="""
def evaluate(run, questions, k=3):
    hit = rec = mr = 0.0
    for q in questions:
        ranked = run(q["question"])
        # YOUR CODE HERE: accumulate hit@k (recall_at_k), recall@k (require_all=True)
        # and mrr(...) for this q
        ...
    n = len(questions)
    return {"hit@%d" % k: hit / n, "recall@%d" % k: rec / n, "mrr": mr / n}

# fake retriever: always returns the same ranking, so the score is hand-checkable
fake_qs = [{"question": "x", "gold_docs": ["b"]},        # hit at rank 2
           {"question": "y", "gold_docs": ["z"]},        # miss
           {"question": "w", "gold_docs": ["a", "z"]}]   # two gold: one at rank 1, one missing
res = evaluate(lambda q: ["a", "b", "c"], fake_qs, k=3)
assert abs(res["hit@3"] - 2 / 3) < 1e-9     # x and w have a gold doc in the top 3
assert abs(res["recall@3"] - 1 / 3) < 1e-9  # only x has every gold doc in the top 3
assert abs(res["mrr"] - 0.5) < 1e-9         # (1/2 + 0 + 1) / 3
print("harness OK:", res)
""",
        solution="""
def evaluate(run, questions, k=3):
    hit = rec = mr = 0.0
    for q in questions:
        ranked = run(q["question"])
        hit += recall_at_k(ranked, q["gold_docs"], k)
        rec += recall_at_k(ranked, q["gold_docs"], k, require_all=True)
        mr += mrr(ranked, q["gold_docs"])
    n = len(questions)
    return {"hit@%d" % k: hit / n, "recall@%d" % k: rec / n, "mrr": mr / n}

# fake retriever: always returns the same ranking, so the score is hand-checkable
fake_qs = [{"question": "x", "gold_docs": ["b"]},        # hit at rank 2
           {"question": "y", "gold_docs": ["z"]},        # miss
           {"question": "w", "gold_docs": ["a", "z"]}]   # two gold: one at rank 1, one missing
res = evaluate(lambda q: ["a", "b", "c"], fake_qs, k=3)
assert abs(res["hit@3"] - 2 / 3) < 1e-9     # x and w have a gold doc in the top 3
assert abs(res["recall@3"] - 1 / 3) < 1e-9  # only x has every gold doc in the top 3
assert abs(res["mrr"] - 0.5) < 1e-9         # (1/2 + 0 + 1) / 3
print("harness OK:", res)
""",
    ),
    md("""
### The payoff — compare the retrievers you built

Set up the dense, BM25 and hybrid (RRF) retrievers over the section chunks. Then
give each retriever a score for all questions together, and for each question
kind. Here the earlier notebooks show their value.
"""),
    code("""
from ragkit.reference import (structure_aware_chunks, chunk_corpus, BM25,
                              reciprocal_rank_fusion, to_doc_ranking)
docs = load_documents()
chunks = chunk_corpus(docs, structure_aware_chunks)
emb = get_embedder()
mat = emb.encode([c.text for c in chunks])
bm = BM25([tokenize(c.text) for c in chunks])
ids = [c.chunk_id for c in chunks]

def dense_run(q):
    order = np.argsort(-(mat @ emb.encode(q)))
    return to_doc_ranking([ids[i] for i in order])

def bm25_run(q):
    ranked = bm.search(q, k=len(chunks))
    return to_doc_ranking([ids[i] for i, _ in ranked])

def hybrid_run(q):
    d = [ids[i] for i in np.argsort(-(mat @ emb.encode(q)))[:10]]
    b = [ids[i] for i, _ in bm.search(q, 10)]
    fused = [cid for cid, _ in reciprocal_rank_fusion([d, b])]
    return to_doc_ranking(fused)

qrels = load_qrels()
runs = {"dense": dense_run, "bm25": bm25_run, "hybrid": hybrid_run}
print(f"{'method':8} {'hit@3':>6} {'recall@3':>9} {'mrr':>6}")
for name, fn in runs.items():
    r = evaluate(fn, qrels, k=3)
    print(f"{name:8} {r['hit@3']:>6.3f} {r['recall@3']:>9.3f} {r['mrr']:>6.3f}")

print("\\nBy question kind (recall@3: every gold doc in the top 3; multihop questions have two):")
for kind in ("lexical", "semantic", "multihop"):
    subset = [q for q in qrels if q["kind"] == kind]
    row = {name: evaluate(fn, subset, k=3)["recall@3"] for name, fn in runs.items()}
    if kind == "multihop":
        assert all(len(q["gold_docs"]) == 2 for q in subset)
        assert all(evaluate(fn, subset, k=3)["recall@3"] <= evaluate(fn, subset, k=3)["hit@3"] for fn in runs.values())
    print(f"  {kind:8}", {k: round(v, 2) for k, v in row.items()})
"""),
    md("""
Read the by-kind table. We expect BM25 to do well on **lexical**, and dense to do
well on **semantic**. We expect **hybrid** to be the most consistent across both.
This is exactly why hybrid is the sensible default.

On recall@3, `multihop` stays difficult for every single-shot retriever. Hit@3
does not show this problem, because one of the two gold documents is sufficient
for a hit. Notebook 06 is about this problem.

### A note on faithfulness

Retrieval metrics do not tell the full story. The retrieved text must actually
*support* a grounded answer. Here is a low-cost proxy: make sure that each answer
sentence has a high-similarity sentence in the context. Systems in production use
an NLI/LLM judge, but the idea is the same.
"""),
    code("""
def faithfulness(answer_text, contexts, emb, thresh=0.5):
    import re
    ctx_sents = [s for c in contexts for s in re.split(r"(?<=[.!?])\\s+", c) if s.strip()]
    ans_sents = [s for s in re.split(r"(?<=[.!?])\\s+", answer_text) if s.strip()]
    if not ans_sents or not ctx_sents:
        return 0.0
    cv = emb.encode(ctx_sents)
    supported = 0
    for s in ans_sents:
        sim = cv @ emb.encode(s)
        supported += int(sim.max() >= thresh)
    return supported / len(ans_sents)

ctx = [docs["expense-policy"]]
grounded = "Meals abroad are reimbursed up to SGD 90 per day."
hallucd = "Meals abroad are reimbursed up to SGD 250 per day and alcohol is included."
print("grounded  :", round(faithfulness(grounded, ctx, emb), 2))
print("hallucin. :", round(faithfulness(hallucd, ctx, emb), 2))
"""),
]


# =========================================================================== #
# 06 — iterative RAG (advanced)
# =========================================================================== #
NB06 = [
    md("""
# 06 · Iterative RAG (advanced)

Single-shot retrieve-then-read fails on **multi-hop** questions. In such a
question, the answer is in two places. The second query only makes sense after you
read the first result. For example, look at this question:

*"Which VPN client does the security policy require, and which OSes does it support?"*

The client name is in one doc, and the OS list is in another.

The solution is a loop:

1. Retrieve.
2. Decide what information you do not have yet.
3. Retrieve again.
4. Stop when you have sufficient information (or when you reach a budget).
"""),
    SETUP,
    code("""
from ragkit.reference import structure_aware_chunks, chunk_corpus, to_doc_ranking
docs = load_documents()
chunks = chunk_corpus(docs, structure_aware_chunks)
emb = get_embedder()
mat = emb.encode([c.text for c in chunks])

def retrieve(query, k=3):
    order = np.argsort(-(mat @ emb.encode(query)))[:k]
    return [chunks[i] for i in order]

# Show the failure: one retrieval for a 2-hop question rarely covers both docs.
q = next(x for x in load_qrels() if x["qid"] == "M1")
single = to_doc_ranking([c.chunk_id for c in retrieve(q["question"], k=3)])
print("question :", q["question"])
print("gold docs:", q["gold_docs"])
print("single-shot retrieved:", single)
"""),
    md("""
### The planner

The decision about the next sub-query is the one step that needs a real LLM. Thus
the planner is pluggable. With an API key, the planner asks the model to make a
decomposition of the question. Offline, it uses a decomposition written by hand
for the demo questions, so the loop still runs. Production has the same shape:
replace the stub with the model.
"""),
    code("""
# Hand-written decompositions so the notebook runs with no API key.
PLANS = {
  "M1": ["approved VPN client required by security policy SEC-011",
         "Meridian Connect supported operating systems macOS Windows Ubuntu"],
  "M2": ["who is paged for a SEV-1 customer-facing outage and how quickly",
         "how is production access granted approval on-call security lead"],
  "M3": ["Scale tier requests per minute rate limit",
         "Scale plan monthly price in SGD"],
}

def plan(question, qid=None):
    if llm.available().startswith(("anthropic", "openai")):
        prompt = ("Break this question into 2 short search queries, one per line, "
                  "each retrieving a different fact needed to answer it.\\n\\nQ: " + question)
        out = llm.complete(prompt, query=question)
        subs = [ln.strip("-- 	").strip() for ln in out.splitlines() if ln.strip()]
        if len(subs) >= 2:
            return subs[:3]
    return PLANS.get(qid, [question])
"""),
    md("""
### Exercise — the iterative loop

Write `iterative_retrieve`. Go through the planned sub-queries in sequence.
Retrieve `k` chunks for each sub-query. Collect the **unique** chunks (remove
duplicates by `chunk_id`). Stop when you reach `budget` sub-queries. Then answer
from all of the chunks that you collected.
"""),
    ex(
        blank="""
def iterative_retrieve(question, qid=None, k=3, budget=3):
    subqueries = plan(question, qid)
    gathered, seen, rounds = [], set(), 0
    for sub in subqueries:
        if rounds >= budget:
            break
        rounds += 1
        # YOUR CODE HERE:
        #   for each chunk from retrieve(sub, k): if its chunk_id is new,
        #   add the chunk to `gathered` and its id to `seen`.
        ...
    return gathered, rounds

gathered, rounds = iterative_retrieve(q["question"], qid="M1", k=3, budget=3)
ids = [c.chunk_id for c in gathered]
assert len(ids) == len(set(ids))          # no duplicates
assert rounds <= 3                         # budget respected
covered = set(to_doc_ranking(ids))
print("rounds:", rounds, "| docs covered:", covered)
assert set(q["gold_docs"]) <= covered      # both hops found (was missed single-shot)
""",
        solution="""
def iterative_retrieve(question, qid=None, k=3, budget=3):
    subqueries = plan(question, qid)
    gathered, seen, rounds = [], set(), 0
    for sub in subqueries:
        if rounds >= budget:
            break
        rounds += 1
        for c in retrieve(sub, k):
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                gathered.append(c)
    return gathered, rounds

gathered, rounds = iterative_retrieve(q["question"], qid="M1", k=3, budget=3)
ids = [c.chunk_id for c in gathered]
assert len(ids) == len(set(ids))          # no duplicates
assert rounds <= 3                         # budget respected
covered = set(to_doc_ranking(ids))
print("rounds:", rounds, "| docs covered:", covered)
assert set(q["gold_docs"]) <= covered      # both hops found (was missed single-shot)
""",
    ),
    md("""
### Answer over the gathered evidence
"""),
    code("""
def build_prompt(query, gathered):
    blocks = [f"[{i}] (source: {c.doc_id})\\n{c.text}" for i, c in enumerate(gathered, 1)]
    ctx = "\\n\\n".join(blocks)
    system = ("Answer using ONLY the sources. Cite [n] after each claim. "
              "The question has multiple parts — answer all of them.")
    return f"{system}\\n\\n=== SOURCES ===\\n{ctx}\\n\\n=== QUESTION ===\\n{query}", [c.text for c in gathered]

prompt, ctxs = build_prompt(q["question"], gathered)
print(llm.complete(prompt, contexts=ctxs, query=q["question"], embedder=emb))
"""),
    md("""
**This is the seed of agentic RAG.** Make the planner a real model. Let the model
select *which tool* to call (vector search, BM25, SQL, web) and *when to stop*.
Add reflection on the results. Make the loop obey a budget. Then you have the loop
that operates modern "deep research" agents.

The mechanics are exactly what you wrote in the exercise. The difficult part is
in the policy and the guardrails.

Where to go next:

- Fine-tune the retriever/reranker on query logs.
- Add a router that sends simple questions down the low-cost single-shot path.
- Do an eval of whole *trajectories* (did it find both docs? in how many steps?),
  not only of final answers.

Part III and Part IV of the primer describe the rest.
"""),
]


NOTEBOOKS = {
    "00_setup_and_corpus": NB00,
    "01_minimal_rag": NB01,
    "02_chunking": NB02,
    "03_hybrid_search": NB03,
    "04_reranking": NB04,
    "05_evaluation": NB05,
    "06_iterative_rag": NB06,
}


def render(cells, solution: bool) -> dict:
    out = []
    for i, c in enumerate(cells):
        cid = f"cell-{i:02d}"                 # stable ids: nbformat 4.5 requires them
        if c["t"] == "md":
            out.append({"cell_type": "markdown", "id": cid, "metadata": {}, "source": c["src"]})
        else:
            src = c.get("sol", c["src"]) if solution else c["src"]
            out.append({"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None,
                        "outputs": [], "source": src})
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def write_nb(nb: dict, path: Path) -> None:
    """Write ``nb`` with the Colab setup cell first, in the injector's JSON layout."""
    nb["cells"].insert(0, _inject.make_cell(path.parent.relative_to(REPO).as_posix()))
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    for name, cells in NOTEBOOKS.items():
        write_nb(render(cells, False), NB_DIR / f"{name}.ipynb")
        write_nb(render(cells, True), SOL_DIR / f"{name}.ipynb")
    print(f"Wrote {len(NOTEBOOKS)} notebooks to {NB_DIR} and solutions to {SOL_DIR}")
