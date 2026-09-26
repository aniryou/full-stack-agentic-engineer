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

RAG has exactly two moving parts: **retrieve** relevant text, then **generate**
an answer conditioned on it. This repo teaches the retrieval half by building it
from scratch, because that is where almost all RAG failures live.

**What you need**

```
pip install -r requirements.txt        # T0: numpy + pytest, no torch
pip install -r requirements-full.txt   # optional: sentence-transformers (and torch) for the real models
```

Without `sentence-transformers`, `ragkit.embed` falls back to a labelled hashing
embedder (lexical, not semantic) and every notebook still runs. With it, the first
embedding call downloads a ~90 MB model (`all-MiniLM-L6-v2`) and then runs offline
on CPU. **Generation** is optional: set `ANTHROPIC_API_KEY` or
`OPENAI_API_KEY` to use a real model, otherwise a deterministic *extractive*
fallback answers from the retrieved text so every notebook still runs.
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

Nine short Markdown docs for a fictional SaaS company, "Meridian". Each has
`##` sections (that structure matters in notebook 02). A small labelled
question set (`qrels`) lets us *measure* retrieval in notebook 05.
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

An **embedding model** maps text to a vector so that *similar meaning → nearby
vector*. "Retrieval" is then just: embed the query, find the nearest document
vectors. Let's see that directly.
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
| 01 | the minimal RAG loop: embed → cosine search → prompt → generate |
| 02 | chunking: why *how* you split decides what you can retrieve |
| 03 | hybrid search: BM25 (exact) + dense (meaning), fused with RRF |
| 04 | reranking: cheap recall, then a cross-encoder for precision |
| 05 | evaluation: hit@k, recall@k and MRR — measure everything above |
| 06 | iterative RAG (advanced): multi-hop questions need >1 retrieval |

Do the exercises in each notebook (fill the `# YOUR CODE HERE` blanks; the
`assert`s check you). Solutions are in `solutions/`.
"""),
]


# =========================================================================== #
# 01 — minimal RAG
# =========================================================================== #
NB01 = [
    md("""
# 01 · The minimal RAG loop

The entire pattern, once:

```
embed the corpus  ─┐
embed the query  ──┤→ cosine similarity → top-k chunks → prompt → generate
```

We start with the crudest possible chunking — **one chunk per document** — so
nothing distracts from the loop. Notebook 02 fixes chunking.
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

Because the rows of `matrix` are L2-normalised, cosine similarity is just the
dot product. Implement `search`: score every chunk against the query, return
the top-`k` as `(chunk, score)` pairs, highest first.
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

Retrieval done. Now put the evidence in front of the model with a numbered
source list and an instruction to answer only from it (this is what makes the
answer *grounded* and *citable*). This part is plumbing, so it's written for
you — read it.
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

Tie it together: retrieve → build the prompt → generate. Use
`llm.complete(...)`, passing `contexts=` and `query=` so the offline fallback
has something to extract, and `embedder=` so it can rank sentences by meaning.
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
That's a working RAG system. Everything after this makes the **retrieve** step
better — because if the right text never makes it into `ctxs`, no amount of
prompting or model quality will save the answer.
"""),
]


# =========================================================================== #
# 02 — chunking
# =========================================================================== #
NB02 = [
    md("""
# 02 · Chunking

Whole-document chunks (nb 01) waste the context window and blur one vector
across many topics. But naive splitting has the opposite failure: it cuts
sentences in half and strips away the heading that gave them meaning.

The key move: **decouple what you match on from what you show the model**.
"""),
    SETUP,
    code("""
docs = load_documents()
sample = docs["security-access"]
print(sample)
"""),
    md("""
### Baseline — fixed-size windows

Split into fixed word windows with a little overlap. Simple, structure-blind.
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

Markdown already tells us the boundaries. Split on `##` headings, and **prepend
the doc title + heading** to each chunk so it is self-describing (a chunk that
says *"Access Control Standard (SEC-011) > Production Access"* retrieves far better than a
bare paragraph that starts with "Access to production...").

Return a list of `(section_heading, chunk_text)` where `chunk_text` starts with
the header line `"<title> > <heading>"`.
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

Precision *and* context: match on a small unit (a sentence), but **return** its
parent section. You search with fine granularity and hand the model enough
surrounding text to answer. The reference `sentence_chunks` implements this —
we import it rather than rebuild it.
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
long. We'll quantify the difference in notebook 05.
"""),
]


# =========================================================================== #
# 03 — hybrid search
# =========================================================================== #
NB03 = [
    md("""
# 03 · Hybrid search — BM25 + dense, fused with RRF

Dense retrieval matches *meaning* but can fumble exact tokens — error codes,
policy IDs, product names. Lexical **BM25** nails those but misses paraphrases.
Real systems run both and fuse the results. We build BM25 from scratch, then
fuse with **Reciprocal Rank Fusion**.
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
    md("""
### Exercise 1 — BM25 scoring

BM25 scores a query term `t` in document `i` as

```
idf(t) · f(t,i)·(k1+1) / ( f(t,i) + k1·(1 − b + b·|d_i|/avgdl) )
```

where `f(t,i)` is the term's frequency in doc `i`, `|d_i|` its length, `avgdl`
the mean length. Fill in that formula (the `idf` and counts are precomputed).
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

Build a dense index over the same chunks and compare the two retrievers on a
**lexical** query (exact code) and a **semantic** query (paraphrase).
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
    md("""
### Exercise 2 — Reciprocal Rank Fusion

RRF combines ranked lists using only **rank position**, so it doesn't care that
BM25 and cosine scores live on different scales:

```
score(d) = Σ_lists 1 / (k + r(d))      # k = 60; r(d) = 1 for the top result of a list
```

(`enumerate` counts from 0, so in code that is `1 / (k + rank + 1)`.)

Implement it, then fuse the dense and BM25 rankings.
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
Hybrid gets the exact-code query *and* the paraphrase. In notebook 05 we'll
confirm with numbers that it beats either method alone across the whole query
set.
"""),
]


# =========================================================================== #
# 04 — reranking
# =========================================================================== #
NB04 = [
    md("""
# 04 · Reranking

First-stage retrievers (dense, BM25) are built to be *cheap* and favour recall:
get the right chunk somewhere in the top 20–100. A **cross-encoder** then reads
each `(query, chunk)` pair *together* and scores relevance far more accurately —
too slow to run over the whole corpus, perfect over a shortlist.

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

`get_cross_encoder()` returns a model with `.predict([(query, passage), ...])`
giving a relevance score per pair. Retrieve `n` candidates, score them, and
return the top `k` chunks by cross-encoder score.
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

Compare the rank of the truly-relevant chunk before vs after reranking. The
cross-encoder typically promotes the exact-answer chunk that first-stage cosine
buried behind topically-similar neighbours.
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
Reranking is usually the single best precision upgrade after hybrid retrieval,
and it's a drop-in: retrieve wide, rerank, keep few.
"""),
]


# =========================================================================== #
# 05 — evaluation
# =========================================================================== #
NB05 = [
    md("""
# 05 · Evaluation

You cannot improve what you don't measure, and "the demo looked good" is not
measurement. With the labelled `qrels` we can score each retriever with two
standard metrics:

* **Hit@k** — is *any* gold document in the top *k*? (Did we fetch something useful?)
* **Recall@k** — are *all* the gold documents in the top *k*? (Did we fetch everything
  the answer needs?) On a one-gold question the two agree; on a two-gold (multihop)
  question hit@k can say 1.0 while half the answer is missing.
* **MRR** — 1/rank of the first gold hit. (How high did it land?)
"""),
    SETUP,
    md("""
### Exercise 1 — the metrics

Implement both over **document-level** rankings (gold labels are per document).
`recall_at_k(..., require_all=False)` is hit@k; with `require_all=True` it is
recall@k. These are pure functions — the asserts pin the exact maths.
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

`evaluate` takes a retrieval function `run(query) -> ranked list of doc_ids` and
averages hit@k, recall@k (every gold document) and MRR over a set of questions. Fill the averaging loop; the
assert uses a fake `run` with a known answer so it's model-independent.
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

Wire up dense, BM25, and hybrid (RRF) over the section chunks, then score them
overall and broken down by question kind. This is where the earlier notebooks
prove themselves.
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
Read the by-kind table: BM25 should shine on **lexical**, dense on **semantic**,
and **hybrid** should be the most consistent across both — which is exactly why
hybrid is the sane default. (`multihop` stays hard for every single-shot
retriever on recall@3 — hit@3 would hide it, because one of the two gold
documents is enough for a hit; that's notebook 06.)

### A note on faithfulness

Retrieval metrics aren't the whole story — a grounded answer must actually be
*supported* by what was retrieved. A cheap proxy: check that each answer
sentence has a high-similarity sentence in the context. Real systems use an
NLI/LLM judge, but the idea is the same.
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

Single-shot retrieve-then-read fails on **multi-hop** questions, where the
answer lives in two places and the second query only makes sense after you've
read the first result. Example: *"Which VPN client does the security policy
require, and which OSes does it support?"* — the client name is in one doc, the
OS list in another.

The fix is a loop: retrieve → decide what's still missing → retrieve again →
stop when you have enough (or hit a budget).
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

Deciding the next sub-query is the one step that wants a real LLM. So the
planner is pluggable: with an API key it asks the model to decompose the
question; offline it uses a hand-written decomposition for the demo questions,
so the loop still runs. (This mirrors production: swap the stub for the model.)
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

Implement `iterative_retrieve`: walk the planned sub-queries, retrieve `k`
chunks for each, accumulate **unique** chunks (dedupe by `chunk_id`), and stop
once you hit `budget` sub-queries. Then answer over everything gathered.
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
**This is the seed of agentic RAG.** Make the planner a real model, let it
choose *which tool* to call (vector search, BM25, SQL, web) and *when to stop*,
add reflection on the results, and enforce a budget — and you have the loop that
powers modern "deep research" agents. The mechanics are exactly what you just
wrote; the sophistication is in the policy and the guardrails.

Where to go next: fine-tune the retriever/reranker on query logs, add a router
that sends simple questions down the cheap single-shot path, and evaluate
whole *trajectories* (did it find both docs? in how many steps?) rather than
just final answers. The primer's Part III and Part IV map the rest.
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
