# minigraphrag

An **educational** reimplementation of the core ideas of Microsoft's
[GraphRAG](https://github.com/microsoft/graphrag) in pure Python + numpy. Like
its sibling [`minifaiss`](../README.md), it exists to be *read*, not to be fast
or accurate: every stage is written plainly, and the two things a real system
adds — **model calls** and **performance/scale machinery** — are each flagged
with a grep-able comment.

> **The library** (`minigraphrag/`) is standard library + numpy only. It runs
> **fully offline and deterministically** out of the box via a rule-based
> `MockLLM` and a hashing embedder, so you can trace the whole pipeline without
> an API key. Point it at a real model (`AnthropicLLM`) for real quality.

```bash
grep -rn "# LLM:"  minigraphrag/   # every place a language/embedding model is called
grep -rn "# PERF:" minigraphrag/   # every place real GraphRAG reaches for scale
```

---

## What is GraphRAG, and what does minigraphrag cover?

Ordinary RAG retrieves loose text chunks by vector similarity and hopes the
answer is in them. That fails on questions whose answer is *spread across* the
corpus ("what are the main themes?") or that need to *connect* facts ("how are X
and Y related?").

GraphRAG fixes this by doing expensive work up front, at **index** time:

1. **Chunk** documents into "text units".
2. **Extract** a knowledge graph from each chunk with an LLM (entities +
   relationships).
3. **Cluster** the graph into a hierarchy of **communities**.
4. **Summarize** every community into a **report** with an LLM.

Then at **query** time it offers two retrieval modes:

- **Local search** — for entity-centric questions: find the query's entities,
  gather their neighbourhood (relationships, source text, community reports),
  and answer from that focused subgraph.
- **Global search** — for whole-corpus questions: map-reduce over the community
  reports (the corpus's auto-generated "table of contents").

minigraphrag keeps every one of those concepts and drops the scale. Each module
mirrors a piece of real GraphRAG:

| minigraphrag                     | real GraphRAG stage            | Core idea |
|----------------------------------|--------------------------------|-----------|
| `chunking.py`                    | `create_base_text_units`       | Slice docs into overlapping text units. |
| `llm.py` · `extract()`           | `extract_graph` (LLM)          | Prompt an LLM for entities + relationships per chunk. |
| `extraction.py`                  | graph merge + description summarize | Merge duplicate nodes/edges across chunks. |
| `graph.py`                       | the entity/relationship tables | Undirected weighted knowledge graph. |
| `community.py`                   | `cluster_graph` (Leiden)       | Cluster into a **hierarchy** of communities (we use Louvain). |
| `summarize.py`                   | `create_community_reports` (LLM) | LLM writes a report per community. |
| `vector_store.py`                | LanceDB / Azure AI Search      | Embed + nearest-neighbour lookup. |
| `local_search.py`               | local search                   | Answer from a focused subgraph. |
| `global_search.py`              | global search                  | Map-reduce over community reports. |
| `index.py`                       | the indexing pipeline          | Orchestrates all of the above; holds artifacts. |
| `prompts.py`                     | the prompt library             | The actual text sent to the model. |

---

## Quickstart

```python
from minigraphrag import Document, GraphRAGIndex, LocalSearch, GlobalSearch

docs = [Document(id="d1", text="Ada Lockwood founded the Meridian Institute ...")]

index = GraphRAGIndex().build(docs)     # offline MockLLM by default; deterministic

# entity-centric question
print(LocalSearch(index).search("Who founded the Meridian Institute?").answer)

# whole-corpus question
print(GlobalSearch(index).search("What are the main themes?").answer)
```

Everything is inspectable on the `index`:

```python
index.stats()                      # counts of every artifact
index.graph.entities               # {id: Entity}
index.communities_at_level(0)      # coarsest communities
index.reports                      # {community_id: CommunityReport}
index.modularity_at_level(0)       # clustering quality
index.save("index.json")           # artifacts serialize to JSON
```

---

## The pipeline, one stage at a time

### 1. Chunking (`chunking.py`)
A sliding word-window with overlap. The overlap gives a relationship that
straddles a boundary a chance to appear intact in some chunk.
**`# PERF:`** real GraphRAG counts *tokens* (tiktoken), not words.

### 2. Extraction (`llm.py` + `extraction.py`)
The LLM reads each chunk and emits **delimited tuples**:

```
("entity"<|>Ada Lockwood<|>PERSON<|>Founder of the Meridian Institute)
##
("relationship"<|>Ada Lockwood<|>Meridian Institute<|>She founded it<|>9)
```

`extraction.py` parses those (the *same* parser handles mock and real output),
then **merges**: the same entity seen in five chunks becomes one node whose
descriptions are combined — and, past a threshold, condensed by another LLM
call. The offline `MockLLM` fakes extraction with proper-noun detection +
sentence co-occurrence, which is enough to build a real graph on tidy text.
**`# LLM:`** extraction and description-merge are model calls.

### 3. The graph (`graph.py`)
Nodes are entities; edges are undirected, weighted relationships (weight = tie
strength, summed across mentions). A hand-rolled adjacency dict, so there is no
graph dependency to hide behind. **`# PERF:`** real scale uses networkx/igraph.

### 4. Communities (`community.py`)
Clusters the graph by **modularity**, using **Louvain** written from scratch:
greedily move nodes between communities to raise modularity, collapse each
community into a super-node, and repeat. Each collapse is a coarser **level**,
so you get a hierarchy (level 0 = coarsest). **`# PERF:`** real GraphRAG uses
**hierarchical Leiden** (graspologic, C-backed, with a refinement phase).

### 5. Community reports (`summarize.py`)
Each community's entities + relationships go to the LLM, which returns a
**JSON** report (title, summary, findings, importance rating). These reports are
what global search reads. **`# LLM:`** report authoring is a model call.

### 6. Embeddings + vector store (`llm.py` + `vector_store.py`)
Entities, text units, and reports are embedded and indexed for nearest-neighbour
lookup. The default embedder is a deterministic **hashing** embedder (lexical
overlap, not meaning); the default store is a brute-force cosine scan.
**`# LLM:`** real embeddings come from a neural model. **`# PERF:`** the store is
where a vector DB / ANN index goes.

### 7. Search (`local_search.py`, `global_search.py`)
Local = vector-search for seed entities → expand to neighbours, relationships,
source text, and reports → LLM answers from that context. Global = LLM map over
each report (relevant points + score) → filter → LLM reduce into one answer.
**`# LLM:`** the answer steps. **`# PERF:`** the global map is embarrassingly
parallel.

---

## Where the LLM lives

The whole point of the `# LLM:` marks is to show *exactly* where intelligence
enters an otherwise mechanical pipeline. There are **six** distinct calls, all
funnelled through `LanguageModel.complete()` — the one method a real backend
implements (see `AnthropicLLM`).

| GraphRAG LLM call            | minigraphrag location                    | `MockLLM` stand-in |
|------------------------------|------------------------------------------|--------------------|
| Entity/relationship extraction | `llm.py` · `extract()` / `heuristic_extract` | proper-noun + co-occurrence rules |
| Description summarization    | `llm.py` · `summarize_descriptions`      | dedup + concatenate |
| Community report authoring   | `summarize.py` · `generate_reports`      | templated JSON from graph stats |
| Global map (points per report) | `global_search.py` (`llm.global_map`)   | lexical overlap + community importance |
| Global reduce (synthesis)    | `global_search.py` (`llm.global_reduce`) | numbered concatenation of points |
| Local answer                 | `local_search.py` (`llm.local_answer`)   | extractive answer from context |
| *(embeddings)*               | `llm.py` · `HashingEmbedding`            | hashed word/char n-grams |

Swap in real Claude calls with one line:

```python
from minigraphrag import AnthropicLLM          # pip install anthropic; ANTHROPIC_API_KEY
index = GraphRAGIndex(llm=AnthropicLLM(model="claude-opus-5")).build(docs)
```

---

## Where performance lives

Every spot where real GraphRAG spends engineering on scale is marked `# PERF:`.

| Real-GraphRAG optimization        | What it does | minigraphrag `# PERF:` location |
|-----------------------------------|--------------|---------------------------------|
| **Token-based chunking**          | Chunk by the model's tokenizer to respect context/cost. | `chunking.py` |
| **Concurrent LLM calls + cache**  | Extraction/report calls run in a bounded worker pool, cached by (prompt, model) so re-indexing is cheap. | `llm.py`, `extraction.py` |
| **Gleaning passes**               | Re-prompt each chunk to catch missed entities. | `extraction.py` |
| **C-backed graph + Leiden**       | igraph/graspologic adjacency and clustering, with a refinement phase and parallelism. | `community.py` (whole module) |
| **Bottom-up report summarization**| Summarize coarse communities from sub-reports so prompts never overflow. | `summarize.py` |
| **Vector database / ANN index**   | LanceDB / Azure AI Search / FAISS instead of a full scan. | `vector_store.py`, `local_search.py` |
| **Parallel map-reduce**           | Global map over reports batched and run concurrently. | `global_search.py` |
| **Checkpointed pipeline (parquet)** | Persist each stage so a failed run resumes without re-calling the LLM. | `index.py` |

The vector store is the cleanest integration point — and it's literally the
sibling package. Back it with `minifaiss` in one line:

```python
from minifaiss import IndexFlatIP
from minigraphrag import MiniFaissVectorStore, GraphRAGIndex
index = GraphRAGIndex(
    vector_store_factory=lambda: MiniFaissVectorStore(IndexFlatIP(512)),
).build(docs)
```

---

## Demo

`demo_graphrag.py` builds a graph from a tiny fictional corpus (a research
institute, an energy company, and a city) and runs both search modes — all
offline.

```bash
./.venv/bin/python demo_graphrag.py
```

Abridged output:

```
GraphRAGIndex(entities=12, relationships=31, communities=2, levels=1)

KNOWLEDGE GRAPH -- entities (by rank = weighted degree)
  Meridian Institute       ORGANIZATION  degree=5 rank=10
  Northwind Corporation    ORGANIZATION  degree=7 rank=10
  Ada Lockwood             PERSON        degree=6 rank=9
  Project Halcyon          CONCEPT       degree=6 rank=9
  ...

COMMUNITIES (hierarchical Louvain; level 0 = coarsest)
  Level 0: 2 communities, modularity Q=0.256
    community 0: Calder Harbor, City of Calder, Elena Vasquez, Harbor Authority, Mayor Torres, ...
    community 1: Ada Lockwood, Calder Energy Summit, Meridian Institute, Professor Chen, Project Halcyon

LOCAL SEARCH
Q: Who is Ada Lockwood and what did she found?
   answer: - Ada Lockwood (PERSON): Ada Lockwood founded the Meridian Institute ...
           Key relationships: Ada Lockwood — Meridian Institute: ...

GLOBAL SEARCH
Q: What are the main themes across these documents?
   answer: Drawing on the community reports, here is what bears on the question:
           1. [Northwind Corporation & Elena Vasquez] This community is organized around ...
           2. [Meridian Institute & Ada Lockwood] This community is organized around ...
```

> **Why a `MockLLM`, not a real model?** So the repo stays dependency-light and
> the pipeline is deterministic and traceable. The mock is deliberately crude —
> its extractor is regex over proper nouns, its answers are extractive — which
> is exactly what makes the `AnthropicLLM` swap illuminating: **the graph, the
> communities, and the search plumbing are unchanged; only the quality of what
> flows through the `# LLM:` points goes up.**

## Tests

```bash
./.venv/bin/python -m pytest tests/test_minigraphrag.py -q
```

Covers each stage (chunking, parsing, graph merge, Louvain, reports) and the
end-to-end index + both search modes, plus a serialization round-trip and the
`minifaiss` vector-store integration.
