# Mistral Primer — Applied Exercises

This file goes with the Mistral section of [`open-weight-llms-primer.md`](open-weight-llms-primer.md) (§4.6). First, try each exercise with no help. Then compare your answer with the worked answers at the end.

---

## Part A — Exercises

### 1. Cost model
A team makes summaries of documents at scale: **400M input tokens** and **40M output tokens** per month. The workload has no agentic tool use and no complex reasoning, but the documents are long.

- (a) Calculate the monthly API cost on Small 4, Large 3 and Medium 3.5.
- (b) Calculate the cost again for the case where 60% of the input is cacheable (a shared system prompt and policy corpus).
- (c) Give the recommendation in one sentence. Then say which facts can change your decision.

### 2. Pipeline routing
Design the Mistral component for each stage of a pipeline for insurance claims:

> 1. Scanned PDFs in six languages.
> 2. Field extraction.
> 3. Validation against a 4,000-page policy corpus.
> 4. An adjudication draft.
> 5. A human review queue.

Name the model or the product for each stage. Also say exactly **which output signal decides what goes to a person**.

### 3. Migration case
In mid-2025, a team built its system on **Magistral Medium 1.2** (reasoning), **Devstral Small 2** (coding agents) and **Pixtral** (vision). A routing layer is in front of the three models.

- (a) What is the migration path today?
- (b) What do they gain?
- (c) Name two things that are not obvious and that break or that you must change.

### 4. Licensing triage
For each use case, say if the licence permits it, and on which model:

| # | Use case |
|---|---|
| a | A commercial TTS voice agent, released to end users |
| b | Fine-tune a model and sell it again as a vertical product |
| c | Content moderation on an air-gapped network |
| d | On-prem code completion in an IDE, with no internet egress |
| e | A vision model on a battery-powered edge device |

### 5. Sovereign architecture
A Singapore bank must obey the MAS guidelines for technology risk and outsourcing. The bank sets this condition: **no personal data leaves Singapore**. It wants document intelligence over its internal credit policies, and also an internal assistant.

- (a) Design the stack.
- (b) **Trap:** why do Mistral Regional Endpoints not solve this on their own?
- (c) Which components must run in-country? Before you approve the design, what do you examine about the infrastructure?

### 6. Explain-it drill
Use five sentences and no notes. Why is Agentic Search better than one-shot RAG? Then name the five tools in order.

---

## Part B — Worked answers

### 1. Cost model

**(a) Uncached** (prices per million tokens):

| Model | Input | Output | **Total/mo** |
|---|---|---|---|
| Small 4 | 400 × $0.15 = $60 | 40 × $0.60 = $24 | **$84** |
| Large 3 | 400 × $0.50 = $200 | 40 × $1.50 = $60 | **$260** |
| Medium 3.5 | 400 × $1.50 = $600 | 40 × $7.50 = $300 | **$900** |

**(b) With 60% cached input** (the price of cached input is 10% of the standard price):

| Model | Uncached in | Cached in | Output | **Total/mo** |
|---|---|---|---|---|
| Small 4 | 160 × $0.15 = $24 | 240 × $0.015 = $3.60 | $24 | **$51.60** |
| Large 3 | 160 × $0.50 = $80 | 240 × $0.05 = $12 | $60 | **$152** |
| Medium 3.5 | 160 × $1.50 = $240 | 240 × $0.15 = $36 | $300 | **$576** |

**(c)** Select Small 4. Its cost is ~11× lower than the cost of Medium 3.5, and the workload needs none of the agentic capability of Medium. Also, `reasoning_effort` gives you more capability on the hard cases, and you do not change models. These facts can change my decision:

- The summaries go to a downstream agent that does multi-step tool calls.
- A quality evaluation on the team's own documents shows that Small 4 fails on long-document coherence. In that case, Large 3, not Medium 3.5, is the next step. Large 3 is still lower in cost than Medium, and Mistral made it for long context.

The point of the exercise: **the high-cost model is rarely the answer, and the low-cost answer is not the smallest model.**

### 2. Pipeline routing

- **Extraction:** OCR 4.1. Its 170 languages include the six languages. It returns bounding boxes, typed block classification and per-page/per-word confidence.
- **Indexing:** Mistral Search Toolkit over the policy corpus. The OCR 4 output goes directly into the ingestion pipeline as citation-ready structured blocks.
- **Validation:** Agentic Search against that index. One-shot RAG fails on exactly this case, because the answer is in a specific clause or table, not in a top-$k$ chunk.
- **Adjudication draft:** Small 4, with `reasoning_effort` set to match the complexity of the claim. Use Medium 3.5 only if the workflow calls multiple tools over a long horizon.
- **Safety:** Use Shieldstral if the outputs go to end users and the deployment is isolated. Use Moderation 2 if the API is acceptable (it is free).

**The signal that routes to a human is OCR 4's confidence score**, per page and per word, together with the block type. This signal is better for routing than a request to the model for its own uncertainty. The calibrated per-word scores of OCR 4 make that request unnecessary.

### 3. Migration case

**(a)** Replace all three with a single model. Use Small 4 if the team is sensitive to cost. Use Medium 3.5 if the coding agents do long-horizon work. Medium 3.5 is the model that replaced Devstral 2 in Mistral's own agent. Delete the routing layer.

**(b)** The team gains these things:

- One deployment instead of three.
- One set of weights for version control and evaluation.
- Fewer GPUs that stay warm.
- No routing logic to maintain, and no errors in routing logic.
- Reasoning becomes a per-request parameter, not a model choice.

**(c) Two non-obvious breaks:**

- **Verbosity changes.** `reasoning_effort="high"` gives approximately the verbosity of the old Magistral. `"none"` gives approximately the chat style of Small 3.2. But do a new benchmark of anything that you adjusted to a specific output length, latency budget or token cost. It is not sufficient to point it at the new model.
- **Infrastructure shape changes.** Devstral Small was a 24B dense model that fits one GPU. Small 4 is a 119B-total, ~6.5B-active MoE (verify, 2026-09). All 119B parameters must be in memory: about 240 GB in BF16 and 120 GB in FP8 (params × bytes, [capacity primer](../gpu-capacity-planning/PRIMER.md)). Thus, in FP8, the model needs two H100s or one H200 before any KV cache. It is possible that a team that self-hosts Devstral Small on one modest GPU must provision its hardware again completely. Or the team can move down to Ministral 3 14B and accept the trade in capability.

### 4. Licensing triage

| # | Verdict |
|---|---|
| a | **No** on open weights. Voxtral TTS is CC BY-NC 4.0. Commercial use needs a separate agreement. |
| b | **Yes** on Apache 2.0: Large 3, Small 4, Ministral 3. Medium 3.5 is Modified MIT. The licence probably permits it, but your legal team must read the modification before you decide. |
| c | **Yes**: Shieldstral, Apache 2.0 and self-hostable. Moderation 2 is Premier and API-only. Thus it fails the air-gap test, although it is free. |
| d | **Not straightforwardly.** Codestral is Premier. To self-host it, you need a negotiated agreement. Small 4 under Apache 2.0 is the alternative with no friction, but it does not have dedicated FIM tuning. |
| e | **Yes**: Ministral 3 3B, Apache 2.0, multimodal, and made for low-resource environments. |

### 5. Sovereign architecture

**(a)** Self-host open-weight models in-country. Use Small 4 or Large 3 for the assistant, and Ministral 3 if the footprint must be small. Run OCR 4 in a single container on the bank's own infrastructure. Build and keep the Search Toolkit index in-country. Use Forge for domain adaptation on the language of the internal credit policies. Forge gives version control, lineage and rollback for the audit trail.

**(b) The trap:** Regional Endpoints let you select **Europe or the US**. Neither is Singapore. For a no-data-leaves-Singapore constraint, regional endpoints are not applicable. You need to self-host, or to use a cloud region inside Singapore. The statement "Just use regional endpoints" has an incorrect idea of what the feature does.

**(c)** These components must be in-country:

- the GPUs that serve the models
- the weights
- the OCR container
- the search index
- the fine-tuning data and Forge's artefacts
- the logs and traces, because the prompts and outputs also contain the personal data.

The options for the GPUs are the bank's own data centre, a hyperscaler region in Singapore, or a sovereign-cloud service.

Before you approve the design, examine these items:

- Every data centre where the service can put the workload. Some regional sovereign clouds span Singapore and adjacent countries.
- The GPU availability and lead time in that region.
- Where support staff and telemetry pipelines can get access to the data.
- If the model licence permits you to self-host (Apache 2.0 for Small 4, Large 3 and Ministral 3; verify).

### 6. Explain-it drill

1. The index only finds candidate documents. The model decides what to examine inside them.
2. Iteration lets the model recover from a weak first retrieval. Thus the model is not stuck with bad chunks.
3. Targeted navigation is better than repeated broad search. It adds accuracy and *decreases* tokens, because precision replaces retries.
4. Then retrieval quality scales with model capability, and your chunking strategy does not set a limit on it.
5. Also, the method is model-agnostic. Thus it improves when models improve, with no infrastructure changes.

Tools: **search, open, navigate, read, grep.**
