# minigraphrag

minigraphrag is an **educational** reimplementation of the core ideas of Microsoft's
[GraphRAG](https://github.com/microsoft/graphrag) in pure Python and numpy. Like its sibling
[`minifaiss`](../README.md), it is for you to *read*. Speed and accuracy are not its purpose. The code shows each
stage in a plain form. A real system adds two things: **model calls** and **performance/scale machinery**. A
grep-able comment marks each of the two.

> **The library** (`minigraphrag/`) uses only the standard library and numpy. With no setup, it runs **fully
> offline and deterministically**, with a rule-based `MockLLM` and a hashing embedder. Thus you can trace the full
> pipeline without an API key. For real quality, connect it to a real model (`AnthropicLLM`).

```bash
grep -rn "# LLM:"  minigraphrag/   # every place a language/embedding model is called
grep -rn "# PERF:" minigraphrag/   # every place real GraphRAG reaches for scale
```

---

## What is GraphRAG, and what does minigraphrag cover?

Ordinary RAG retrieves loose text chunks by vector similarity. It then expects, with no guarantee, that the answer is
in these chunks. This method fails on two types of question:

- A question whose answer is *spread across* the corpus ("what are the main themes?").
- A question that must *connect* facts ("how are X and Y related?").

GraphRAG solves this problem. It does the high-cost work first, at **index** time:

1. **Chunk** documents into "text units".
2. **Extract** a knowledge graph from each chunk with an LLM (entities and relationships).
3. **Cluster** the graph into a hierarchy of **communities**.
4. **Summarize** every community into a **report** with an LLM.

Then, at **query** time, it gives two retrieval modes:

- **Local search** is for entity-centric questions. It finds the entities of the query and collects their
  neighbourhood (relationships, source text, community reports). Then it answers from that focused subgraph.
- **Global search** is for whole-corpus questions. It does a map-reduce over the community reports (the
  auto-generated "table of contents" of the corpus).

minigraphrag keeps all of these concepts, but not the scale. Each module matches a part of the real GraphRAG:

| minigraphrag                     | real GraphRAG stage            | Core idea |
|----------------------------------|--------------------------------|-----------|
| `chunking.py`                    | `create_base_text_units`       | Divides the documents into text units that overlap. |
| `llm.py` · `extract()`           | `extract_graph` (LLM)          | Asks an LLM for the entities and relationships in each chunk. |
| `extraction.py`                  | graph merge + description summarize | Merges duplicate nodes/edges across chunks. |
| `graph.py`                       | the entity/relationship tables | An undirected weighted knowledge graph. |
| `community.py`                   | `cluster_graph` (Leiden)       | Clusters the graph into a **hierarchy** of communities (we use Louvain). |
| `summarize.py`                   | `create_community_reports` (LLM) | The LLM writes a report for each community. |
| `vector_store.py`                | LanceDB / Azure AI Search      | Embeds, then does a nearest-neighbour lookup. |
| `local_search.py`               | local search                   | Answers from a focused subgraph. |
| `global_search.py`              | global search                  | Does a map-reduce over the community reports. |
| `index.py`                       | the indexing pipeline          | Runs all the other stages in this table and holds the artifacts. |
| `prompts.py`                     | the prompt library             | The actual text that goes to the model. |

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

You can examine everything on the `index`:

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
The chunker uses a sliding word-window with overlap. Because of the overlap, it is possible that a relationship
across a boundary appears complete in one of the chunks.
**`# PERF:`** the real GraphRAG counts *tokens* (tiktoken), not words.

### 2. Extraction (`llm.py` + `extraction.py`)
The LLM reads each chunk and writes **delimited tuples**:

```
("entity"<|>Ada Lockwood<|>PERSON<|>Founder of the Meridian Institute)
##
("relationship"<|>Ada Lockwood<|>Meridian Institute<|>She founded it<|>9)
```

`extraction.py` parses these tuples. The *same* parser reads the mock output and the real output. Then the module
**merges** the results. If the same entity occurs in five chunks, it becomes one node, and that node combines the
descriptions. Past a threshold, one more LLM call makes the combined descriptions shorter.

The offline `MockLLM` simulates the extraction with proper-noun detection and sentence co-occurrence. This is
sufficient to build a real graph on clean text.
**`# LLM:`** the extraction and the description-merge are model calls.

### 3. The graph (`graph.py`)
The nodes are entities. The edges are undirected, weighted relationships (the weight is the tie strength, summed
across mentions). The graph is an adjacency dict written by hand, thus no graph dependency hides the mechanism.
**`# PERF:`** a system at real scale uses networkx/igraph.

### 4. Communities (`community.py`)
This module clusters the graph by **modularity**. It uses **Louvain**, written from scratch, in these steps:

1. Move nodes between communities with a greedy rule, to increase modularity.
2. Collapse each community into a super-node.
3. Do steps 1 and 2 again.

Each collapse gives a coarser **level**, thus you get a hierarchy (level 0 = coarsest). **`# PERF:`** the real
GraphRAG uses **hierarchical Leiden** (graspologic, C-backed, with a refinement phase).

### 5. Community reports (`summarize.py`)
The entities and relationships of each community go to the LLM. The LLM returns a **JSON** report (title, summary,
findings, importance rating). Global search reads these reports. **`# LLM:`** a model call writes each report.

### 6. Embeddings + vector store (`llm.py` + `vector_store.py`)
The module embeds and indexes the entities, text units and reports for nearest-neighbour lookup. The default
embedder is a deterministic **hashing** embedder. It measures lexical overlap, not meaning. The default store is a
brute-force cosine scan.

**`# LLM:`** real embeddings come from a neural model. **`# PERF:`** a vector DB / ANN index goes in the place of
the store.

### 7. Search (`local_search.py`, `global_search.py`)
Local search has three steps:

1. A vector search finds the seed entities.
2. The search adds the neighbours, relationships, source text and reports of these entities.
3. The LLM answers from that context.

Global search has three steps:

1. The LLM does a map over each report (relevant points and a score).
2. A filter operates on the results.
3. The LLM does a reduce of the results into one answer.

**`# LLM:`** the answer steps. **`# PERF:`** the global map is embarrassingly parallel.

---

## Where the LLM lives

The main purpose of the `# LLM:` marks is to show *exactly* where intelligence enters a pipeline that is otherwise
mechanical. There are **six** different calls. All of them go through `LanguageModel.complete()`. This is the one
method that a real backend implements (see `AnthropicLLM`).

| GraphRAG LLM call            | minigraphrag location                    | `MockLLM` stand-in |
|------------------------------|------------------------------------------|--------------------|
| Entity/relationship extraction | `llm.py` · `extract()` / `heuristic_extract` | proper-noun and co-occurrence rules |
| Description summarization    | `llm.py` · `summarize_descriptions`      | dedup and concatenation |
| Community report authoring   | `summarize.py` · `generate_reports`      | templated JSON from graph stats |
| Global map (points per report) | `global_search.py` (`llm.global_map`)   | lexical overlap and community importance |
| Global reduce (synthesis)    | `global_search.py` (`llm.global_reduce`) | numbered concatenation of points |
| Local answer                 | `local_search.py` (`llm.local_answer`)   | extractive answer from context |
| *(embeddings)*               | `llm.py` · `HashingEmbedding`            | hashed word/char n-grams |

To use real Claude calls, change one line:

```python
from minigraphrag import AnthropicLLM          # pip install anthropic; ANTHROPIC_API_KEY
index = GraphRAGIndex(llm=AnthropicLLM(model="claude-opus-5")).build(docs)
```

---

## Where performance lives

A `# PERF:` mark shows each location where the engineers of the real GraphRAG do work for scale.

| Real-GraphRAG optimization        | What it does | minigraphrag `# PERF:` location |
|-----------------------------------|--------------|---------------------------------|
| **Token-based chunking**          | It chunks by the tokenizer of the model, to stay in the context/cost limits. | `chunking.py` |
| **Concurrent LLM calls + cache**  | The extraction/report calls run in a bounded worker pool. A cache with the key (prompt, model) makes a second index run low-cost. | `llm.py`, `extraction.py` |
| **Gleaning passes**               | It sends each chunk to the model again to find the entities that it missed. | `extraction.py` |
| **C-backed graph + Leiden**       | igraph/graspologic adjacency and clustering, with a refinement phase and parallelism. | `community.py` (whole module) |
| **Bottom-up report summarization**| It summarizes coarse communities from their sub-reports, so the prompts never overflow. | `summarize.py` |
| **Vector database / ANN index**   | LanceDB / Azure AI Search / FAISS, in place of a full scan. | `vector_store.py`, `local_search.py` |
| **Parallel map-reduce**           | It batches the global map over the reports and runs the batches in parallel. | `global_search.py` |
| **Checkpointed pipeline (parquet)** | It writes the output of each stage to storage. Thus a failed run resumes and does not call the LLM again. | `index.py` |

The vector store is the cleanest integration point. The sibling package does exactly the job of a vector store. Put
`minifaiss` behind the store with one line:

```python
from minifaiss import IndexFlatIP
from minigraphrag import MiniFaissVectorStore, GraphRAGIndex
index = GraphRAGIndex(
    vector_store_factory=lambda: MiniFaissVectorStore(IndexFlatIP(512)),
).build(docs)
```

---

## Demo

`demo_graphrag.py` builds a graph from a small fictional corpus (a research institute, an energy company, and a
city). Then it runs the two search modes. All of this runs offline.

```bash
./.venv/bin/python demo_graphrag.py
```

The output, shortened:

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

> **Why a `MockLLM`, not a real model?** The mock keeps the repo dependency-light, and it makes the pipeline
> deterministic and traceable. The mock is crude on purpose. Its extractor is a regex over proper nouns, and its
> answers are extractive. This is exactly why the change to `AnthropicLLM` teaches you much: **the graph, the
> communities and the search infrastructure code stay the same. Only the quality of the data that goes through the
> `# LLM:` points increases.**

## Tests

```bash
./.venv/bin/python -m pytest tests/test_minigraphrag.py -q
```

The tests cover each stage (chunking, the parser, graph merge, Louvain, reports). They also cover the end-to-end index
and the two search modes, a serialization round-trip and the `minifaiss` vector-store integration.
