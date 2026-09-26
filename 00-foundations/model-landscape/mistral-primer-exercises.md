# Mistral Primer — Applied Exercises

Companion to the Mistral section of [`open-weight-llms-primer.md`](open-weight-llms-primer.md) (§4.6). Attempt each cold, then check against the worked answers at the end.

---

## Part A — Exercises

### 1. Cost model
A team runs document summarisation at scale: **400M input tokens** and **40M output tokens** per month. No agentic tool use, no complex reasoning, but long documents.

- (a) Compute monthly API cost on Small 4, Large 3 and Medium 3.5.
- (b) Recompute assuming 60% of input is cacheable (shared system prompt and policy corpus).
- (c) State the recommendation in one sentence, and say what would change your mind.

### 2. Pipeline routing
Design the Mistral component at each stage of an insurance claims pipeline:

> scanned PDFs in six languages → field extraction → validation against a 4,000-page policy corpus → adjudication draft → human review queue

Name the model or product per stage, and say specifically **which output signal decides what goes to a human**.

### 3. Migration case
A team built in mid-2025 on **Magistral Medium 1.2** (reasoning) + **Devstral Small 2** (coding agents) + **Pixtral** (vision), with a routing layer in front.

- (a) What is the migration path today?
- (b) What do they gain?
- (c) Name two non-obvious things that break or need rework.

### 4. Licensing triage
For each, state whether it's permissible, and on which model:

| # | Use case |
|---|---|
| a | Commercial TTS voice agent shipped to end users |
| b | Fine-tune a model and resell it as a vertical product |
| c | Content moderation on an air-gapped network |
| d | On-prem code completion in an IDE, no internet egress |
| e | Vision model on a battery-powered edge device |

### 5. Sovereign architecture
A Singapore bank under MAS technology risk and outsourcing guidelines requires that **no personal data leaves Singapore**. It wants document intelligence over internal credit policies plus an internal assistant.

- (a) Design the stack.
- (b) **Trap:** why do Mistral Regional Endpoints not solve this on their own?
- (c) Which components must run in-country, and what would you check about the infrastructure before signing off the design?

### 6. Explain-it drill
In five sentences, no notes: why does Agentic Search beat one-shot RAG? Then name the five tools in order.

---

## Part B — Worked answers

### 1. Cost model

**(a) Uncached** (prices per million tokens):

| Model | Input | Output | **Total/mo** |
|---|---|---|---|
| Small 4 | 400 × $0.15 = $60 | 40 × $0.60 = $24 | **$84** |
| Large 3 | 400 × $0.50 = $200 | 40 × $1.50 = $60 | **$260** |
| Medium 3.5 | 400 × $1.50 = $600 | 40 × $7.50 = $300 | **$900** |

**(b) With 60% cached input** (cached priced at 10% of standard):

| Model | Uncached in | Cached in | Output | **Total/mo** |
|---|---|---|---|---|
| Small 4 | 160 × $0.15 = $24 | 240 × $0.015 = $3.60 | $24 | **$51.60** |
| Large 3 | 160 × $0.50 = $80 | 240 × $0.05 = $12 | $60 | **$152** |
| Medium 3.5 | 160 × $1.50 = $240 | 240 × $0.15 = $36 | $300 | **$576** |

**(c)** Small 4 — it's ~11× cheaper than Medium 3.5 for a workload that needs none of Medium's agentic capability, and `reasoning_effort` gives you headroom on the hard cases without changing models. What would change my mind: if the summaries feed a downstream agent doing multi-step tool calls, or if quality evaluation on their own documents shows Small 4 failing on long-document coherence — in which case Large 3, not Medium 3.5, is the next step, since it's still cheaper than Medium and built for long context.

The point of the exercise: **the expensive model is rarely the answer, and the cheap answer isn't the smallest model.**

### 2. Pipeline routing

- **Extraction:** OCR 4.1 — 170 languages covers the six, and it returns bounding boxes, typed block classification and per-page/per-word confidence.
- **Indexing:** Mistral Search Toolkit over the policy corpus; OCR 4 output feeds the ingestion pipeline directly as citation-ready structured blocks.
- **Validation:** Agentic Search against that index — this is exactly the case one-shot RAG fails, since the answer sits in a specific clause or table rather than a top-*k* chunk.
- **Adjudication draft:** Small 4 with `reasoning_effort` scaled to claim complexity; Medium 3.5 only if the workflow calls multiple tools over a long horizon.
- **Safety:** Shieldstral if outputs reach end users and the deployment is isolated; Moderation 2 if API is acceptable (it's free).

**The signal that routes to a human is OCR 4's confidence score** — per-page and per-word — combined with block type. It is a better routing signal than asking the model for its own uncertainty, which OCR 4's calibrated per-word scores make unnecessary.

### 3. Migration case

**(a)** Collapse all three onto a single model. Small 4 if cost-sensitive, Medium 3.5 if the coding agents are doing long-horizon work — Medium 3.5 is what replaced Devstral 2 in Mistral's own agent. Delete the routing layer.

**(b)** One deployment instead of three, one set of weights to version and evaluate, fewer GPUs held warm, no routing logic to maintain or get wrong, and reasoning becomes a per-request parameter rather than a model choice.

**(c) Two non-obvious breaks:**
- **Verbosity changes.** `reasoning_effort="high"` approximates old Magistral verbosity and `"none"` approximates Small 3.2 chat style — but anything tuned to a specific output length, latency budget or token cost needs re-benchmarking, not just re-pointing.
- **Infrastructure shape changes.** Devstral Small was a 24B dense model that fits one GPU; Small 4 is a 119B-total, ~6.5B-active MoE (verify, 2026-09). All 119B parameters must sit in memory: about 240 GB in BF16 and 120 GB in FP8 (params × bytes, [capacity primer](../gpu-capacity-planning/PRIMER.md)), so two H100s or one H200 in FP8 before any KV cache. A team self-hosting Devstral Small on one modest GPU may need to re-provision entirely — or drop to Ministral 3 14B and accept the capability trade.

### 4. Licensing triage

| # | Verdict |
|---|---|
| a | **No** on open weights. Voxtral TTS is CC BY-NC 4.0; commercial use needs a separate agreement. |
| b | **Yes** on Apache 2.0 — Large 3, Small 4, Ministral 3. Medium 3.5 is Modified MIT: probably fine, but legal must read the modification before you commit. |
| c | **Yes** — Shieldstral, Apache 2.0 and self-hostable. Moderation 2 is Premier and API-only, so it fails the air-gap test despite being free. |
| d | **Not straightforwardly.** Codestral is Premier; self-hosting needs a negotiated agreement. Small 4 under Apache 2.0 is the frictionless alternative, at the cost of dedicated FIM tuning. |
| e | **Yes** — Ministral 3 3B, Apache 2.0, multimodal, built for low-resource environments. |

### 5. Sovereign architecture

**(a)** Open-weight models self-hosted in-country — Small 4 or Large 3 for the assistant, Ministral 3 if the footprint must be small. OCR 4 in a single container on the bank's own infrastructure. Search Toolkit index built and held in-country. Forge for domain adaptation on internal credit policy language, with versioning, lineage and rollback for the audit trail.

**(b) The trap:** Regional Endpoints let you choose **Europe or the US**. Neither is Singapore. For a no-data-leaves-Singapore constraint, regional endpoints are irrelevant — you need self-hosting or a cloud region inside Singapore. "Just use regional endpoints" misreads the feature.

**(c)** In-country: the GPUs serving the models, the weights, the OCR container, the search index, the fine-tuning data and Forge's artefacts, and the logs and traces, because prompts and outputs contain the personal data too. Options for the GPUs are the bank's own data centre, a hyperscaler region in Singapore, or a sovereign-cloud offering. Before signing off, check: every data centre the offering may place the workload in (some regional sovereign clouds span Singapore and neighbouring countries), GPU availability and lead time in that region, where support staff and telemetry pipelines can reach the data, and whether the model licence allows self-hosting (Apache 2.0 for Small 4, Large 3 and Ministral 3; verify).

### 6. Explain-it drill

The index only identifies candidate documents; the model decides what to actually inspect inside them. Iteration lets it recover from a weak first retrieval instead of being stuck with bad chunks. Targeted navigation beats repeated broad search — it adds accuracy while *reducing* tokens, because precision replaces retries. Retrieval quality then scales with model capability rather than being capped by your chunking strategy. And it's model-agnostic, so it improves as models improve without infrastructure changes.

Tools: **search, open, navigate, read, grep.**

