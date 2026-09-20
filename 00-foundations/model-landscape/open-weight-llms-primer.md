# Open-Weight LLMs: A Primer

**State of play as of 29 August 2026.** This field moves in weeks. Specs, dates and benchmark figures below reflect the best public sources at time of writing, with primary sources where they exist and vendor claims flagged as such. Verify any number before it goes into a client deck or a procurement document.

---

## 1. The short version

- **"Open weight" means the trained parameters are downloadable and runnable on your own hardware.** It does not mean the training data, code or recipe are public. Very few capable models are open in that fuller sense.
- **Chinese labs still set the open frontier** — Moonshot (Kimi K3), DeepSeek (V4), Alibaba (Qwen 3.8), Z.ai (GLM-5.2), MiniMax (M3) — but 2026 brought a genuine Western re-entry: Google moved Gemma 4 to Apache 2.0, NVIDIA shipped Nemotron 3 Ultra with data and recipes, Thinking Machines released Inkling, Meta returned with Muse Glimmer, and Mistral has a new open family in early access.
- **The capability gap is roughly one release cycle.** Epoch AI measures about four months, or ~8 points on its capability index, for 2026 to date. On Artificial Analysis the top closed model (Claude Opus 5, 63) leads the top open model (Kimi K3, ~57) by six points.
- **The gap is jagged, not uniform.** Open models are at or near parity on agentic coding and computer use, and behind on hard knowledge reasoning (HLE, GPQA, ARC-AGI).
- **August 2026 broke two patterns.** Alibaba opened a Max-class flagship for the first time (Qwen3.8-2.4T-A95B), and Meta committed to opening Muse Spark 1.2. The "closed frontier tier, open tier one generation behind" model is no longer the only game.
- **The most consequential tier for practitioners is 27–35B dense.** Qwen3.8-27B scores 52 on Artificial Analysis in a checkpoint that runs on one 24 GB GPU.
- **Enterprise adoption is paradoxical.** Menlo Ventures measured open-source models falling from 19% to 11% of enterprise LLM usage in 2025 even as capability converged. The constraint is governance, TCO and operations, not model quality.
- **Licenses are diverging again.** Apache 2.0 and MIT remain the centre of gravity, but several 2026 flagships shipped under bespoke licenses with revenue thresholds (Qwen3.8-Max, MiniMax M3, Kimi K3). Read the actual file.
- **Policy has split.** EU AI Act enforcement powers activated 2 August 2026; the US excluded open-weight models from its voluntary frontier testing program on 4 August 2026.

---

## 2. What "open weight" actually means

Openness is a spectrum. The useful distinctions:

| Tier | What is released | Typical license | Examples |
|---|---|---|---|
| Closed / API-only | Nothing downloadable | Terms of service | Claude Opus 5, GPT-5.6 Sol, Gemini 3.x, Grok 4.6, Muse Spark 1.2 (for now), Mistral Medium 3.5 |
| Open weights, conditional | Weights under a bespoke license with scale, revenue or field-of-use conditions | Qwen3.8-Max License, MiniMax Community License, Modified MIT, Llama Community License | Qwen3.8-2.4T-A95B, MiniMax M3, Kimi K3, Llama 4 |
| Open weights, permissive | Weights under an OSI-approved license | Apache 2.0, MIT | DeepSeek V4, GLM-5.2, Qwen3.8-27B, Gemma 4, gpt-oss, Mistral Large 3, Inkling, Muse Glimmer |
| Open source (per OSI's definition) | Weights + code + enough data information to substantially recreate | Permissive license plus data disclosure | Few frontier-class models qualify |
| Fully open | Weights + data + code + checkpoints + recipe | Apache 2.0 or similar | AI2 OLMo 3, NVIDIA Nemotron 3, Swiss Apertus, HF SmolLM |

Points that matter in practice:

- **OSI's Open Source AI Definition (Oct 2024)** requires data information, code and parameters. Most "open" LLMs fail on data. "Open weight" is the honest term.
- **What is usually withheld even by open labs:** the pretraining corpus, RL environments and reward models, the full post-training recipe, and sometimes capability that exists in the closed sibling. Alibaba's open 2.4T checkpoint is text-only with thinking forced on, while the hosted `qwen3.8-max` API adds vision, video and a default 1M context. Meta has said it will keep some capabilities — notably cyber-offensive code generation — out of its open releases.
- **Transparency is falling while capability rises.** Stanford's Foundation Model Transparency Index dropped from 58 to 40, with less disclosure of data, parameter counts and compute.
- **Variants per family:** base, instruct, thinking/reasoning (usually with an effort dial), coder, vision, omni, guard/safety, embedding. Distilled small models are a large share of the ecosystem.

---

## 3. How we got here

| When | What | Why it mattered |
|---|---|---|
| 2019 | Staged GPT-2 release | Set the release-norms debate |
| 2022 | BLOOM, OPT, Pythia | Research-grade open models |
| Feb 2023 | LLaMA leaks | llama.cpp, Alpaca, LoRA — the ecosystem forms |
| Jul–Dec 2023 | Llama 2, Mistral 7B, Mixtral 8x7B | Commercially usable open weights; MoE goes mainstream |
| 2024 | Llama 3.1-405B, Qwen 2.5, Gemma, DeepSeek-V2 (MLA) and V3 (FP8, ~$5.6M run); OSI publishes OSAID | 405B briefly matched the closed frontier; Chinese labs take the efficiency lead |
| Jan 2025 | DeepSeek-R1 under MIT | RL reasoning reproduced in the open; market shock |
| 2025 | Qwen3, Kimi K2 (1T, Muon), GLM-4.5–4.7, gpt-oss (Aug), Llama 4 disappoints and Behemoth is shelved, Mistral Large 3 + Ministral 3 (Dec), OLMo 3, Granite 4, Nemotron 3 Nano | Chinese labs dominate cadence; Meta reported to be going closed |
| Feb–Mar 2026 | Qwen 3.5 (397B-A17B); Nemotron Coalition (16 Mar); Mistral Small 4, Leanstral, Forge | Coalition-style open development begins |
| Apr 2026 | Gemma 4 Apache 2.0 (2 Apr); Meta's closed Muse Spark (8 Apr); Qwen 3.6 open tier (22 Apr); DeepSeek V4 (24 Apr) | Google drops its custom license; DeepSeek ships 1.6T/1M under MIT |
| May–Jun 2026 | Qwen 3.7 closed (May); MiniMax M3 (1 Jun); Nemotron 3 Ultra (4 Jun); Kimi K2.7 Code (12 Jun); GLM-5.2 (13 Jun) | Three frontier open coding models in a fortnight |
| Jul 2026 | Inkling (15 Jul); Kimi K3 (16 Jul, weights 27 Jul); Inkling-Small (30 Jul); Mistral open family in early access; Muse Spark 1.1 (9 Jul) | Largest open release ever; a credible US open-frontier entrant |
| Aug 2026 | Muse Spark 1.2 + Muse Code (5 Aug); US exempts open weights from testing (4 Aug); EU enforcement powers (2 Aug); Muse Glimmer (10 Aug); Qwen3.8-2.4T-A95B (12 Aug); DeepSeek V4-Pro GA (13 Aug); Qwen3.8-27B (14 Aug) | Alibaba opens a Max-class model; Meta returns to open weights; policy lines drawn |

---

## 4. The families, vendor by vendor

Closed reference points for calibration: Claude Opus 5 (AA Intelligence Index 63), Claude Fable 5 (62), GPT-5.6 Sol (61), Grok 4.6 (61), Muse Spark 1.2 (57).

### 4.1 Moonshot AI — Kimi (China)

**Current:** Kimi K3 (announced 16 Jul 2026, weights 27 Jul, arXiv:2607.24653). 2.8T total / **104B active**, native vision, 1M context, Modified MIT. Kimi K2.7 Code (12 Jun) remains a ~1T coding specialist with MoonViT vision.

**Position:** the strongest open model on most aggregate indices, and the largest open-weight release to date. Moonshot has held the open scale frontier for nine of the past twelve months.

**Innovations that matter:**
- **Kimi Delta Attention (KDA)** — a fixed-size-state linear attention with a forget gate, from the *Kimi Linear* line of work. K3 interleaves three KDA layers with one Gated MLA layer per block: cheap linear mixing for most of the sequence work, periodic full-capacity attention to preserve global interaction. This is the first and largest public model built primarily on linear attention, and it is the strongest signal yet that softmax attention will not stay universal.
- **Attention Residuals (AttnRes)** — each module can selectively retrieve representations from the embedding, its own block and all preceding blocks, rather than only reading the immediately previous layer. Shortens the information path in very deep networks.
- **Stable LatentMoE** — 896 routed experts with 16 active (~56x sparsity, up from K2's 384/8). Shared experts run at full hidden width; routed experts operate in a narrower latent space via down-projection before dispatch and up-projection after aggregation. Stabilised with SiTU-GLU activation, RMSNorm on routed experts and Quantile Balancing.
- **Per-head Muon** optimizer; **quantization-aware training from the SFT stage** (not post-hoc), shipped in MXFP4 — so the quantized checkpoint degrades far less than a post-training quant would.
- Net effect: roughly **2.5x the scaling efficiency of K2** — compute converted to capability, not just parameters added.

**Deployment reality:** ~594 GB quantized, ~1.56 TB at BF16. Cluster-scale only. Most teams will consume it through a hosted endpoint.

**License watch:** Modified MIT — attribution obligations kick in above roughly 100M monthly users or $20M monthly revenue.

### 4.2 DeepSeek (China)

**Current:** V4-Pro (1.6T / ~49B active) and V4-Flash (284B / ~13B active), released 24 Apr 2026 under **MIT**, both 1M context with 384K max output, **text-only**. V4-Pro reached GA on 13 Aug 2026 (checkpoint V4-Pro-0813). Technical report: arXiv:2606.19348.

**Position:** the price-performance and efficiency benchmark for the whole field. V4-Pro-Max is reported at 80.6% SWE-bench Verified — the top open-weight entry — and a Codeforces rating of 3206, the first open model to match a closed system on competitive programming.

**Innovations that matter:**
- **Manifold-Constrained Hyper-Connections (mHC)** (arXiv:2512.24880, co-authored by CEO Liang Wenfeng). Hyper-connections widen the residual stream into multiple parallel streams, but unconstrained mixing matrices destroy training stability — DeepSeek measured signal amplification above 3000x and catastrophic divergence at 27B. mHC projects those matrices onto the Birkhoff polytope using Sinkhorn–Knopp, holding amplification to ~1.6x for ~6.7% training overhead. This is the single most-cited architectural idea to come out of the open ecosystem this year.
- **Hybrid attention: Compressed Sparse Attention (CSA) + Heavily Compressed Attention (HCA)**, replacing V3.2's Multi-head Latent Attention. At 1M tokens, V4-Pro uses roughly **27% of the single-token inference FLOPs and 10% of the KV cache** of V3.2. That is the number that drives the pricing.
- **Muon optimizer**; 32T training tokens; separate domain specialists trained then merged.
- **Engram** (conditional memory via scalable lookup) was widely expected in V4 and is **not** in the final architecture — a useful correction to a lot of early coverage.
- Open kernels: DeepGEMM / MegaMoE for fine-grained expert-parallel communication–computation overlap.

**Economics:** V4-Pro at ~$0.435 in / $0.87 out per million; V4-Flash at ~$0.14 / $0.28 — roughly 30x below closed-frontier output pricing. V4-Flash is the practical self-host target; V4-Pro fits a single 8-GPU B200 node but needs cluster economics to serve at competitive latency.

**Operational note:** the legacy `deepseek-chat` and `deepseek-reasoner` aliases retired 24 Jul 2026. Also, most serverless hosts quantize V4 activations to FP8, which moves outputs away from the reference weights — pin your host's precision if reproducibility matters.

### 4.3 Alibaba — Qwen (China)

**Current:** the broadest ladder in the ecosystem, and the family that changed posture in August.
- **Qwen3.8-Max** (3 Aug 2026): 2.4T total / ~95B active sparse MoE, 1M context, text + image + video, $2 / $6 per million on the hosted API.
- **Qwen3.8-2.4T-A95B** (12 Aug 2026): the open-weight Max checkpoint — **the first Max-class Qwen ever made downloadable**. 512 experts (10 routed + 1 shared), Qwen3.5-family hybrid (Gated DeltaNet + MoE + Gated Attention). Custom **Qwen3.8-Max License**, not Apache.
- **Qwen3.8-27B** (14 Aug 2026): 27.78B dense vision-language model under **Apache 2.0**. 64 layers, hidden dim 5120, hybrid layout of 48 Gated DeltaNet linear-attention layers and 16 full-attention layers, multi-token prediction, `reasoning_effort` dial (xhigh default / medium / low). Native 262,144-token context, extensible to ~1M with YaRN. Text, image and video input.
- Still current: Qwen3.6-27B and Qwen3.6-35B-A3B, Qwen 3.5-397B-A17B, and the full 0.6B–14B ladder.

**Position and the reversal:** through the 3.7 generation Alibaba kept the frontier closed (3.7-Max, 3.7-Plus, a robotics VLA) and the open tier a generation behind. The 3.8 generation reversed that within ten days of launch. Note the asymmetry that remains: the open 2.4T checkpoint is **text-only, with thinking forced on and native 262K context**, while the hosted API keeps vision, video, optional thinking and a default 1M window.

**Why Qwen3.8-27B is the important one:** it scores **52 on the Artificial Analysis Intelligence Index — up 14 points over Qwen3.6-27B on an architecturally identical model**. That delta is post-training alone, and it is the clearest demonstration this year that RL and data recipes, not parameter count, are where the remaining headroom is. It runs on ~17 GB at 4-bit (an 18 GB Ollama download) on a single 3090 or 4090. Community tuning reports ~114 tok/s single-user on a power-limited 3090 and ~1,000 tok/s aggregate across 64 parallel streams.

**Benchmarks, with the asterisk:** Alibaba's own card reports SWE-bench Pro 61.7, Terminal-Bench 2.1 73.0 (up from 63.4), OSWorld-Verified 84.3 (up from 63.9), LiveCodeBench v6 90.3, IFBench 79.5 — beating Claude Opus 4.6 Max on 16 of 24 rows. It loses on GPQA Diamond, Terminal-Bench and Humanity's Last Exam (30.8 vs 40.0). Several of those benchmarks are in-house or "corrected" versions. Independent results have since landed: #1 open-weight on Harvey's Legal Agent benchmark, #9 overall on Code Arena's WebDev leaderboard, and #1 open-weight (#7 overall) on Arena.ai's Image-to-WebDev board.

**Known quirk:** the default `xhigh` reasoning effort dramatically over-thinks simple prompts. Set the dial deliberately.

**License watch:** Qwen3.8-27B is clean Apache 2.0. The 2.4T checkpoint is not — it is an MIT-style grant with an attribution rider above 100M MAU or $20M monthly revenue, plus a **separate paid license for model-as-a-service or AI-work-assistant businesses above $50M trailing-twelve-month revenue including affiliates**, with purely internal use carved out. That last clause is the one to route past legal.

### 4.4 Z.ai / Zhipu — GLM (China)

**Current:** GLM-5.2 (13 Jun 2026), ~744–753B MoE, **MIT**, 1M context, text-only, with explicit reasoning-effort presets.

**Position:** the pragmatic default for self-hosting. Not the top scorer, but the combination of an unambiguous MIT license, the most mature quantization tooling of the Chinese flagships, and heavy real-world routing traffic makes it the one teams actually run. One gateway reported GLM-5.2 carrying ~4.5x the traffic of the higher-scoring Muse Spark 1.2 over the same week.

**What changed from 5.1:** GLM-5.1 was widely judged architecturally weak; 5.2 moved to Tier A on independent coding benchmarks in a single generation, with large gains on coding and agentic tasks and on long-horizon persistence. It is also a math and reasoning outlier relative to its overall index position. Nous Research integrated it into Hermes Agent within days of release.

**Numbers:** SWE-bench Pro 62.1, Terminal-Bench 2.1 81.0 (vendor). Artificial Analysis v4.1.1 scores it 53 — briefly the highest open-weights score on that board. LLM Stats open leaderboard: 46.8 overall. Roughly $1.40 / $4.40 per million on hosted endpoints.

**Corporate context:** Zhipu was the first Chinese AI lab to list publicly, in Hong Kong in January 2026, and its market cap has since run to record levels.

### 4.5 MiniMax (China)

**Current:** MiniMax M3 (1 Jun 2026), 428B total / ~23B active, up to 1M context (512K guaranteed minimum), natively multimodal from step 0 of training — text, image and video in — and able to operate a desktop.

**Position:** the only open-weight model pairing frontier agentic coding (80.5% SWE-bench Verified, vendor-reported) with native multimodality. For UI automation and screenshot-to-code at low cost it has no open peer. $0.30 / $1.20 per million at MiniMax's "permanent 50% off" rate.

**Innovation that matters — MiniMax Sparse Attention (MSA)** (arXiv:2606.13392): a GQA backbone with block-level selection over **real, uncompressed** key-values, rather than the compressed latent state MLA uses. MiniMax's argument is that this partitions the KV more precisely than DSA or MoBA and avoids MLA's compression-versus-precision trade-off. Reported effect: ~15.6x faster decoding and ~9.7x faster prefill at 1M context versus M2, with per-token compute at 1M around 1/20th of M2's. The MSA kernels are published separately under MIT.

**License correction worth flagging:** M3 is **not** MIT. It ships under the custom **MiniMax Community License** (`license: other`, `license_name: minimax-community`), which requires prominently displaying "Built with MiniMax M3" for commercial use and requires **separate prior written authorization from MiniMax for organizations earning over $20M USD annually** from products built on it. The permissive MIT license on the MSA kernel repository does not extend to the weights. This is a real compliance gate, not a formality.

### 4.6 Mistral AI (France)

**Current lineup:** Mistral Large 3 (2 Dec 2025) — 675B total / 41B active MoE, Apache 2.0, still the largest Apache-licensed MoE from a Western lab. Mistral Small 4 (16 Mar 2026) folds reasoning (Magistral), vision (Pixtral) and coding (Devstral) into one ~24B model. Ministral 3 at 3B / 8B / 14B, all Apache 2.0 — the 14B reasoning variant hits 85% on AIME 2025. Plus specialists: Devstral 2 (code), Voxtral (audio/TTS), Leanstral 1.5 (Lean 4 formal proofs), Mistral OCR, and Shieldstral (a 3B open-weights multimodal safety classifier that accepts plain-language policies at inference time and runs on a single 16 GB GPU). Mistral Medium 3.5 is closed.

**What's coming:** Mensch has confirmed a new open-weight family — described as "fat but sparse" MoE — in **early access since July 2026**, with a broader release expected. Parameter count, benchmarks and license terms remain undisclosed. Separately, the **first Nemotron Coalition model is a base model co-developed by Mistral and NVIDIA on DGX Cloud, to be open-sourced on completion and to underpin Nemotron 4**.

**Position:** the sovereignty play. Mistral is the only European frontier lab betting on open weights at scale, is a signatory to the EU GPAI Code of Practice, and has the compliance posture EU-regulated buyers want. Commercially: ARR above $400M in early 2026 (from ~$20M a year earlier), targeting $1B by year end; €1.7B Series C led by ASML at a €11.7B valuation, with later discussions reported above $23B; a €4B data-centre buildout.

**Honest read on capability:** Large 3 is a strong non-reasoning model (MMLU-Pro ~73.1, MATH-500 ~93.6 on independent evaluation) but scores far lower on reasoning-heavy benchmarks (~40% AIME 2025, ~44% GPQA Diamond) and is slow for its class at ~38 tok/s. The summer release is what will settle whether Mistral is at the open frontier or one tier below it.

**License watch:** most releases are Apache 2.0, but not all — a few (some audio and older research models) use CC-BY-NC or the Mistral Research License. Check per model.

### 4.7 Google DeepMind — Gemma (US)

**Current:** Gemma 4 (2 Apr 2026) in five sizes — E2B (2.3B effective), E4B (4.5B effective), a 12B unified multimodal, a 26B MoE with ~4B active, and a 31B dense. Built on Gemini 3 research. 128K context on the small variants, 256K on the larger. Vision across the board; audio on E2B and E4B. 140+ languages.

**The headline is the license, not the benchmarks.** Gemma 4 is the first Gemmaverse release under the **OSI-approved Apache 2.0** license, replacing the custom Gemma Terms of Use with their updatable prohibited-use policy and flow-down obligations. There is no compete clause: you can build a product that competes directly with Google's own. If you evaluated and rejected an earlier Gemma on legal grounds, this is the version to re-examine.

**Capability:** at launch the 31B dense ranked #3 and the 26B MoE #6 among open models on the Arena text leaderboard, with the 31B at ~1452 Elo — outperforming models up to 20x its size on intelligence-per-parameter.

**Strategy:** this is a funnel, not charity. If the most capable open models carry Google's DNA, Google Cloud becomes the natural scaling destination. Gemma also anchors sovereign deployments — state licensing automation in Ukraine, Project Navarasa across India's 22 official languages.

### 4.8 Meta (US)

**The arc:** Llama 4 (Apr 2025) underwhelmed, Behemoth was shelved, and through late 2025 Meta was widely reported to be abandoning open weights for a closed frontier model codenamed Avocado. Meta Superintelligence Labs under Alexandr Wang shipped that as **Muse Spark** (8 Apr 2026, closed), then **Muse Spark 1.1** (9 Jul), then **Muse Spark 1.2** (5 Aug) alongside **Muse Code**, a co-trained terminal coding agent. Muse Spark 1.2 scores **57 on Artificial Analysis** (rescored up from 54 in index v4.1.1 — the largest single increase in that update), with a 1M context at $1.25 / $4.25 per million.

**The return to open:** on 10 August Meta released **Muse Glimmer** — a ~30B dense multimodal model under **Apache 2.0**, distilled from Muse Spark 1.2, built for always-on local agent workloads, running offline on a single 24 GB consumer GPU at under 20 GB quantized. Zuckerberg paired it with a 6,500-word essay ("The Future is for Everyone"), a $1B community fund for data-centre neighbours, a call for lower US barriers on open-source AI, and a commitment to **open the weights of Muse Spark 1.2 itself**.

**Status as of late August: Muse Spark 1.2 weights have not shipped.** Wang said "coming soon" on 10 August; there is no repository, license file or date. If it lands, it would be the strongest US open-weight model by a distance and a genuine rival to the Chinese tier. Until then it is a promise.

**Also worth knowing:** there is **no Llama 5** — Meta's model site now leads with Muse and lists Llama 4 behind it; third-party forecasts push Llama 5 to 2027. Llama 4 Maverick remains the last Llama flagship, and much of the "Llama 5 spec sheet" content in search results is fabricated. Meta's contributor API tier ($0.10 / $0.20) buys the discount with permission to train on your prompts and completions — read that before routing private code through it.

### 4.9 NVIDIA — Nemotron (US)

**Current:** the Nemotron 3 family — Nano (~31.6B total / 3.2B active), Super (~120B / 12B active), and **Ultra (4 Jun 2026, ~550B total / up to ~55B active)**, plus Nemotron 3.5 Lightning (Aug 2026) for high-volume always-on agents and specialist models for robotics, AVs, drug discovery and voice.

**Position:** the most genuinely open of the big-lab releases. NVIDIA publishes **weights, training data and recipes**, with technical reports sufficient to recreate the models — the closest thing to OSI-style openness at this scale. That is a materially different governance proposition for regulated buyers.

**Innovations that matter:**
- **Hybrid Mamba-2 + Transformer MoE.** Mamba-2 state-space layers give linear-time complexity over sequence length, which is what makes a 1M-token context economical for long-running agents rather than merely possible.
- **LatentMoE** — compresses tokens into a low-rank latent space before routing, enabling roughly 4x as many expert specialists at the same inference cost.
- **Multi-Token Prediction** — predicts several future tokens per forward pass, improving chain-of-thought coherence and giving built-in speculative decoding at serve time.
- **NVFP4 four-bit pretraining** on Blackwell, with claimed ~5x throughput efficiency and ~30% cost reduction versus the best open alternatives.

**Ecosystem proof points** (NVIDIA-reported, but specific and attributable): LangChain tuned its Deep Agents harness for Nemotron 3 Ultra — prompts, tools and middleware only, no retraining — and reached top agent accuracy among open models at ~10x lower cost per run than leading closed alternatives. Arcee AI post-trained on Blackwell to ~$0.90 per million output tokens, ~20x cheaper than comparable closed frontier models, ranking second on PinchBench while staying fully open weight. Harvey post-trained Ultra on its legal benchmark and matched leading closed models at ≥10x lower cost per run. YTL AI Labs post-trained a Nemotron for Malay. Nemotron 3 Nano Omni was measured by Artificial Analysis at 323 tok/s — the fastest model on the BenchLM board clearing its evidence thresholds.

**Nemotron Coalition** (announced 16 Mar 2026): Black Forest Labs, Cursor, LangChain, Mistral AI, Perplexity, Reflection AI, Sarvam and Thinking Machines Lab, co-developing open frontier models on DGX Cloud. First deliverable is the Mistral–NVIDIA base model that will underpin Nemotron 4. Contributions span multimodal (Black Forest), real-world coding benchmarks (Cursor) and agentic/long-horizon evaluation (LangChain).

### 4.10 Thinking Machines Lab (US)

**Current:** **Inkling** (15 Jul 2026) — 975B total / ~41B active MoE, **Apache 2.0**, trained on 45 trillion tokens of text, image, audio and video, reasoning natively across all four, 1M context, controllable thinking effort. Built in nine months on NVIDIA GB300 NVL72 systems. **Inkling-Small** (30 Jul) — 276B total / 12B active, also Apache 2.0.

**Why it matters:** this is the most credible US open-weight entrant of 2026 and the clearest articulation of an alternative business model. Mira Murati's lab explicitly states Inkling is "not the strongest overall model available today, open or closed" and does not monetize the model at all — revenue comes from **Tinker**, its fine-tuning platform (customers include Bridgewater). The bet is that "good enough + fully customizable + free" beats "smartest but locked up."

**The interesting result:** Inkling-Small **beats its own parent** on reasoning and agentic rows — HLE 31.6 vs 29.7, SWE-bench Verified 80.2 vs 77.6, Terminal-Bench 2.1 64.7, Toolathlon Verified 54.4 vs 45.5, ARC-AGI-2 40.1 vs 36.5, GPQA Diamond 89.5, AIME 2026 95.5 — while losing badly on factuality (SimpleQA Verified 20.6 vs 43.9; AA-Omniscience −9.0 vs 2.1). It was distilled from an Inkling checkpoint and then given two further weeks of agentic-coding RL. That is a clean, public demonstration of the current frontier trade: **RL on a smaller student buys agentic capability and costs you world knowledge.** Plan your routing accordingly.

**Safety posture, unusually explicit for an open release:** trained to an internal safety spec across modalities with commissioned external testers; internal and external evaluations for CBRN, cyber and loss-of-control; attention to sycophancy, vulnerable users and manipulation. The lab claims the strongest built-in safeguards of any open-weights model it compared on FORTRESS. It also trained Inkling to answer directly on censorship-prone topics, and Cognition's Propaganda and Censorship Eval found strong censorship non-compliance — a deliberate contrast with Chinese-origin models.

**Deployment:** BF16 needs ≥600 GB aggregated VRAM (4x B300 or 8x H200); the NVFP4 checkpoint drops the floor to 180 GB. Inkling itself needs close to 2 TB. Available via Tinker, Databricks, Together, Fireworks, Modal, Baseten, and a free rate-limited OpenRouter endpoint.

### 4.11 OpenAI (US)

**Current:** gpt-oss-120b (116.8B total / 5.1B active) and gpt-oss-20b (21B / 3.6B), released Aug 2025 under **Apache 2.0**, plus gpt-oss-safeguard (Oct 2025). 128K context, low/medium/high reasoning effort, attention sinks, shipped natively **MXFP4-quantized** so the 120b fits one 80 GB GPU and the 20b runs in ~16 GB.

**Position:** it became infrastructure rather than a headline. MLCommons added gpt-oss-120b as a standard **MLPerf Inference v6.0 benchmark in March 2026** — the hardware industry now treats it as a reference workload. It remains a common default for zero-budget self-hosted agent stacks.

**The notable absence:** there is no gpt-oss-2 and no announced successor, a year on. OpenAI's open contribution now looks like a one-off strategic release plus a durable safety-methodology contribution — the **worst-case fine-tuning evaluation** (adversarially post-training your own model to see what a malicious actor could extract) is the closest thing the field has to an emerging norm for open releases.

### 4.12 The rest of the field, briefly

- **AI2 — OLMo 3** (7B / 32B, incl. Think variants, Apache 2.0): the reference fully-open model. Data (Dolma 3), code, intermediate checkpoints and recipe all published. The right choice when reproducibility is the requirement.
- **IBM — Granite 4** (small dense and hybrid Mamba-2 MoE, Apache 2.0): enterprise governance emphasis, ISO 42001-certified pipeline. A sensible default for regulated, low-risk workloads.
- **Microsoft — Phi-4** (4B–15B, MIT): small-model specialists for reasoning and multimodal.
- **xAI**: Grok 4.5 / 4.6 are closed and near the frontier (4.6 at AA 61); xAI's stated policy is to open prior generations, as it did with Grok 2.5 in Aug 2025.
- **Cohere — Command A / A+**: strong enterprise RAG models, but CC-BY-NC — non-commercial unless you buy a license.
- **Other Chinese labs:** ByteDance (Seed-OSS), Tencent (Hunyuan), Baidu (ERNIE 4.5, Apache 2.0), Ant Group (Ling/Ring), Meituan (LongCat), Xiaomi (MiMo).
- **Sovereign and regional:** TII Falcon-H1 (UAE), LG EXAONE and Naver HyperCLOVA X (Korea), Sarvam (India), Apertus (Switzerland), AI Singapore's SEA-LION (Southeast Asian languages). This is where most APAC enterprise open-model demand actually sits.
- **Builders on top:** Arcee AI (post-training specialists), Nous Research (Hermes agents), Reflection AI (US open-frontier startup, Coalition member), Liquid AI (LFM2 edge), Hugging Face (SmolLM).
- **Anthropic** releases no open weights.

---

## 5. The innovation map: what actually changed in 2026

Five threads run across every family above.

**1. Attention is being rebuilt around the KV cache, not the FLOPs.** At 1M context the cache, not the weights, is the binding memory constraint. Four distinct bets are now in production:
- *Compression* — DeepSeek's MLA → CSA + HCA lineage. Cheapest, with a precision cost MiniMax explicitly criticises.
- *Block selection on uncompressed KV* — MiniMax's MSA. Preserves precision, more machinery.
- *Linear attention with periodic full attention* — Kimi's KDA + Gated MLA, Qwen's Gated DeltaNet + full-attention layers. Fixed-size recurrent state; now proven at 2.8T.
- *State-space hybrids* — Mamba-2 layers in Nemotron 3, Granite 4, Falcon-H1. Linear-time by construction.

There is no convergence yet, and the choice directly determines your serving cost. It also determines whether a GGUF fallback preserves the advantage: MiniMax M3's sparse attention advantage largely disappears on the dense-attention local inference path most people without a GPU rack will actually take.

**2. MoE sparsity is climbing fast, and stability is the bottleneck.** Kimi went from 384 experts / 8 active to 896 / 16 (~56x sparsity) in one generation; DeepSeek V4-Pro activates ~3% of 1.6T; Qwen3.8-Max ~4% of 2.4T. Every lab that pushed sparsity had to solve activation explosions and expert-utilisation collapse — hence Stable LatentMoE, Quantile Balancing, SiTU-GLU (Moonshot), LatentMoE with low-rank routing (NVIDIA), and shared-plus-fine-grained expert designs generally.

**3. The residual stream itself is now a design surface.** DeepSeek's mHC widens it under a manifold constraint; Kimi's AttnRes lets layers reach back across depth. Both attack the same problem — one narrow residual path bottlenecking information flow in very deep, very wide models — and both were validated at scale in the same quarter. Expect more here.

**4. Quantization moved from post-hoc to native.** gpt-oss shipped in MXFP4; Nemotron 3 pretrains in NVFP4; Kimi K3 does quantization-aware training from the SFT stage. "Full precision" and "quantized" are converging, which quietly removes one of the standard objections to self-hosting.

**5. Post-training is where the remaining headroom is.** The cleanest evidence of the year: **Qwen3.8-27B gained 14 Artificial Analysis points over Qwen3.6-27B on an architecturally identical model**, at the cost of roughly 2x the tokens per task. Inkling-Small beat its own 3.5x-larger teacher on agentic rows after two extra weeks of coding RL. The ladder is now standard — SFT → RL with verifiable rewards (the GRPO lineage from DeepSeek-R1) → agentic RL in tool and computer-use environments — and it is exposed to users as reasoning-effort dials and thinking-context preservation. Open **agentic RL environments and reward models remain the last major piece of the stack that is still mostly closed.**

---

## 6. Where the frontier is now

**The closed frontier (late August 2026), on Artificial Analysis Intelligence Index:**

| Model | Lab | AA Index | Notes |
|---|---|---|---|
| Claude Opus 5 (24 Jul) | Anthropic | 63 | Also leads the Agentic Index at 55.3 and ARC-AGI-3 at 30.16%; $5 / $25 |
| Claude Fable 5 | Anthropic | 62 | #1 Arena text (1508.6), HLE 53.3%, AA-Omniscience 40, ARC-AGI-1 leader |
| GPT-5.6 Sol | OpenAI | 61 | ARC-AGI-2 92.5% verified; Terminal-Bench 2.1 88.8% |
| Grok 4.6 | xAI | 61 | Back in the frontier pack |
| Muse Spark 1.2 | Meta | 57 | Weights promised, not shipped |
| Gemini 3.7 Flash | Google | — | The price-performance outlier: ARC-AGI-2 84.6% at ~$0.25/task |

**The open frontier:**

| Model | Lab | Size | License | Standing |
|---|---|---|---|---|
| Kimi K3 | Moonshot | 2.8T / 104B | Modified MIT | #1 open. AA ~57; LLM Stats open leaderboard 55.4; #3 overall on AA at one point, behind only Fable 5 and GPT-5.6 Sol; #1 on Frontend Code Arena; 3rd on LM Arena Agent (0.142 vs Opus 5's 0.176) |
| Qwen3.8-2.4T-A95B | Alibaba | 2.4T / 95B | Custom | Newest Max-class open checkpoint; text-only vs the API |
| DeepSeek V4-Pro | DeepSeek | 1.6T / 49B | MIT | 80.6% SWE-bench Verified (V4-Pro-Max), Codeforces 3206; best price-performance in the field |
| Inkling | Thinking Machines | 975B / 41B | Apache 2.0 | 77.6% SWE-bench Verified; strongest US open entry; explicitly not chasing #1 |
| GLM-5.2 | Z.ai | ~750B | MIT | AA 53; the practical self-host default; highest real routing traffic |
| MiniMax M3 | MiniMax | 428B / 23B | Community | 80.5% SWE-bench Verified; only open model with native multimodality + 1M + frontier coding |
| Inkling-Small | Thinking Machines | 276B / 12B | Apache 2.0 | 80.2% SWE-bench Verified — beats its own parent on agentic rows |
| Qwen3.8-27B | Alibaba | 27.8B dense | Apache 2.0 | AA 52 on a single 24 GB GPU. The best intelligence-per-gigabyte available |

**How wide is the gap, measured three ways:**
- **Epoch AI:** since January 2026, the best open models lag the closed frontier by an average of **four months, or ~8 ECI points** (90% CI 7–11) — wider than the three-month average measured over 2023–2025. Under a stricter comparison rule, ~six months. Epoch's July index put Kimi K3 at 156 against GPT-5.6 Sol at 162, with **overlapping confidence intervals**.
- **Artificial Analysis:** Opus 5 at 63 vs Kimi K3 at ~57 — six points. Of 170–185 tracked models, roughly 95 are open weight.
- **Stanford AI Index (March 2026 snapshot):** 1,503 Arena points for the leading closed model vs 1,454 for the leading open model — a 3.3% gap, against 0.5% in August 2024. US–China gap at the top: ~2.7%.

**Two reasons the real gap is probably wider than these numbers.** Epoch notes that open models appear to hill-climb public benchmarks more aggressively and do relatively worse on private evaluations, and that closed labs withhold their strongest models — so open is being compared against the *published* frontier, not the actual one.

**The shape of the frontier, honestly:**

- **Open is at parity or ahead** on agentic coding (multiple open models above 80% SWE-bench Verified), computer use and desktop control (Qwen3.8-Max reportedly at 86.1 OSWorld-Verified, ahead of GPT-5.6 Sol Max and Fable 5), instruction following, long-context retrieval, extraction, classification and most multilingual work.
- **Open is behind** on hard knowledge reasoning and abstract problem solving: Humanity's Last Exam (Qwen3.8-27B 30.8 vs Opus 4.6 Max 40.0; Fable 5 at 53.3), GPQA Diamond, ARC-AGI-2/3, research synthesis, and reliability on genuinely novel tasks.
- **Open has won on price and forced the closed tier to respond.** Open API list prices run ~8x below closed on average, DeepSeek's ~30x below on output tokens. On 30 July OpenAI cut Terra 20% and Luna 80% citing serving-cost improvements — the steepest cut of the year from a US lab, and a direct answer to the Chinese open tier.
- **Usage and procurement still diverge.** Chinese models were reported at ~61% of OpenRouter traffic in June 2026; Menlo's enterprise survey put Chinese open models at ~1% of enterprise LLM API usage. Both are true — different populations.

**The three structural shifts to take away:**

1. **Scale has bifurcated.** The open flagships are 0.4–2.8T MoE models that essentially nobody self-hosts — they are consumed through hosted endpoints, where the open/closed distinction collapses to licensing and provenance rather than deployment. The models people actually run are 4–35B, and that tier got dramatically better in 2026.
2. **The tiering pattern is breaking down, unevenly.** Alibaba opened a Max-class model; Meta committed to opening its flagship; NVIDIA and Thinking Machines open everything. Google, Mistral and Alibaba still run a closed top tier. Treat open weights as a strategic lever any lab can withdraw, not a principle — Alibaba's 3.7 generation shipped entirely closed, one generation before it opened a 2.4T flagship.
3. **America re-entered, but China still sets the pace.** Muse Glimmer, Inkling, Nemotron 3 Ultra and gpt-oss are real US open contributions, and Nemotron is the most transparent release from any large lab. But the top of the open leaderboard is Moonshot, DeepSeek, Alibaba, Z.ai and MiniMax, and the architectural innovations everyone is now copying — MLA/CSA, mHC, KDA, LatentMoE, MSA, Muon at scale — came predominantly from labs operating under GPU export constraints.

---

## 7. Licensing: what to actually check

| License | Commercial use | Notable terms | Used by |
|---|---|---|---|
| Apache 2.0 | Yes | Patent grant; attribution; no copyleft | Qwen3.8-27B and open Qwen tier, most Mistral, Gemma 4, gpt-oss, Muse Glimmer, Inkling, Granite, OLMo, ERNIE 4.5 |
| MIT | Yes | Minimal; no patent grant | DeepSeek V3/R1/V4, GLM-5.x, Phi |
| Modified MIT (Moonshot) | Yes | Attribution above ~100M MAU or ~$20M monthly revenue | Kimi K2 / K3 |
| Qwen3.8-Max License | Yes, with limits | MIT-style + attribution rider above 100M MAU / $20M monthly revenue; **separate paid license for MaaS or AI-work-assistant businesses above $50M TTM revenue incl. affiliates**; internal use carved out | Qwen3.8-2.4T-A95B |
| MiniMax Community License | Yes, with limits | "Built with MiniMax M3" attribution; **prior written authorization required above $20M annual revenue** | MiniMax M3, M2.7 |
| Open model license w/ data + recipes | Yes | Covers weights and training data; permissive | Nemotron 3 family |
| Llama Community License | Yes, with limits | 700M-MAU threshold; "Built with Llama" naming; AUP; Llama 4 restricts EU-domiciled multimodal use | Llama 2–4 |
| Gemma Terms of Use | Yes, with limits | Updatable prohibited-use policy; flow-down | Gemma 1–3 (Gemma 4 is Apache 2.0) |
| CC-BY-NC 4.0 | No | Non-commercial only | Cohere Command A; some Mistral audio releases |
| Mistral Research License | No | Research only; commercial via Mistral | Older Mistral models |

**Diligence checklist**

1. **Scale and revenue thresholds** — MAU, monthly revenue, TTM revenue including affiliates, and what happens when you cross them. Three of 2026's flagship open models have these.
2. **Field-of-use restrictions** — the Qwen3.8-Max MaaS clause is the clearest example; if you resell inference, read it first.
3. **Acceptable-use policies** — can the provider update them unilaterally, and do they bind your customers?
4. **Attribution and naming** — "Built with X" in product names or docs.
5. **Outputs** — ownership, and whether you may train other models on them.
6. **Derivatives** — can you ship fine-tuned weights, and under what license?
7. **Patent grant** — Apache 2.0 has one; MIT does not.
8. **Jurisdiction and export controls** — geographic carve-outs (Llama 4's EU restriction) and your own export exposure.
9. **Data you add** — fine-tuning datasets carry their own licenses; synthetic data from closed APIs may carry terms restricting competing-model training.
10. **Hygiene** — roughly 70% of scanned Hugging Face repositories carried no license tag in May 2026. A missing tag is not permission. Check the LICENSE file on the exact checkpoint, and note that a permissively licensed kernel repository (MiniMax's MSA) does not license the weights.
11. **Indemnity and support** — cloud catalogs (Bedrock, Vertex, Azure AI Foundry) and vendors like Mistral, IBM and NVIDIA sell indemnified versions. That is usually what compliance actually wants.

---

## 8. Running, adapting and operating open models

### Memory math

- **Weights:** total parameters × bytes per parameter (BF16 = 2, FP8 = 1, INT4/FP4 ≈ 0.5–0.6 with scales), plus 10–20% overhead. For MoE, **all** experts must be resident even though few are active.
- **KV cache:** context × layers × KV heads × head dim × bytes. MLA, sparse attention, linear and SSM layers shrink it dramatically. At 1M context it dominates.
- **Worked examples (approximate):** gpt-oss-20b ~16 GB · Qwen3.8-27B ~17 GB at 4-bit (18 GB Ollama download), ~28 GB FP8, ~56 GB BF16 before KV · Muse Glimmer <20 GB quantized · Gemma 4 31B ~18–20 GB at 4-bit · gpt-oss-120b ~65 GB MXFP4 (one 80 GB GPU) · Inkling-Small 180 GB NVFP4 / ≥600 GB BF16 · DeepSeek V4-Flash ~150–170 GB at 4-bit · DeepSeek V4-Pro fits one 8-GPU B200 node at low precision · Qwen3.8-2.4T above 400 GB even at aggressive 1-bit quants · Kimi K3 ~594 GB quantized, ~1.56 TB BF16 — multi-node only.

### Deployment tiers

| Tier | Hardware | What runs well |
|---|---|---|
| Laptop / edge | 8–32 GB RAM, NPU or consumer GPU | Gemma 4 E2B/E4B, Ministral 3, Qwen 3.6-4B, Phi |
| Workstation | 24–32 GB GPU, or 128 GB unified memory (DGX Spark, Strix Halo, M-series Max) | **Qwen3.8-27B, Gemma 4 31B, Muse Glimmer, gpt-oss-20b**; 100B-class MoE on 128 GB |
| Single server | 1–8 × 80–192 GB datacenter GPUs | gpt-oss-120b, V4-Flash, Nemotron Super, Mistral Large 3, Inkling-Small (NVFP4) |
| Cluster | Multi-node H100/H200/B200/GB300 | V4-Pro, GLM-5.2, Kimi K3, Nemotron 3 Ultra, Inkling, Qwen3.8-Max |
| Hosted open endpoints | None of your own | Everything above — with the caveat that host quantization and versioning vary |

### The serving stack

- **Engines:** vLLM and SGLang (day-0 support for most major releases — DeepSeek V4 had both on release day), TensorRT-LLM, llama.cpp / Ollama / LM Studio / MLX locally.
- **Cost levers:** paged and prefix KV caching, continuous batching, disaggregated prefill/decode, speculative decoding (MTP heads), FP8/FP4 kernels, expert parallelism.
- **Formats:** safetensors for weights (never load pickle from untrusted sources), GGUF for llama.cpp, AWQ/GPTQ/MXFP4/NVFP4/Q4_K_M quants. Sparse-attention models need their documented serving flags — a 428B MoE with MSA will not serve well on generic settings.
- **Hosted providers:** OpenRouter, Together, Fireworks, Groq, Cerebras, Baseten, DeepInfra, Lambda, Morph, plus hyperscaler catalogs and regional clouds. Most speak OpenAI-compatible and increasingly Anthropic-compatible APIs, so agent harnesses swap models via environment variables.

### The adaptation ladder (cheapest first)

1. **Prompt and harness tuning** — frequently the biggest single win. LangChain reached top open-model agent accuracy on Nemotron 3 Ultra by tuning prompts, tools and middleware alone. Note also that Meta benchmarks Muse Spark 1.2 inside its co-trained Muse Code harness — model and harness are not separable in vendor charts.
2. **Retrieval and tool grounding** — same as closed, but you can co-locate the model with the data.
3. **PEFT** — LoRA/QLoRA via TRL, Unsloth, Axolotl, LLaMA-Factory. Hours on one GPU for a 27B model.
4. **Full post-training and RL** — NeMo, TRL, verl-style stacks; or managed routes (Mistral Forge, Thinking Machines' Tinker). Harvey reached frontier-class legal accuracy on Nemotron at ≥10x lower cost per run; Arcee hit ~$0.90 per million output tokens.
5. **Distillation** — compress a large open teacher into a task-specific student. Inkling-Small is the public worked example, including its factuality cost.

### Operating discipline

- **Governance:** record model version, quantization format, serving engine version and license text in production. "We run GLM-5.2" stops being a complete answer once three quants and two point releases are in play. Forrester's Model Openness Framework (Apr 2026) and Gartner's AI TRiSM material are usable scaffolding.
- **Safety layers:** pair the model with a guard classifier (Shieldstral, gpt-oss-safeguard, Llama Guard) plus prompt-injection and output controls. Built-in refusals are fine-tunable and therefore not a control you can lean on alone.
- **Lifecycle:** nothing gets deprecated under you — but you own upgrades, regression testing and serving-stack patching.
- **Cost reality:** don't self-host on rented GPUs unless you keep them saturated. Below that, a hosted open endpoint is cheaper. The OECD published a break-even model in 2026 worth using as a TCO template.

---

## 9. Why enterprises choose open weights — and why they often don't

**Drivers:** data residency and sovereignty; cost at volume; latency, edge and offline operation; customisation (where open models most often *beat* closed for a specific task); predictability and independence from silent deprecations; auditability of a fixed checkpoint.

**Frictions:** operational burden and capex; capability lag on the hardest tasks and worse private-benchmark behaviour than leaderboards suggest; license diligence — where most deals actually stall; provenance and security; support and indemnity usually requiring a vendor, which erodes part of the cost advantage.

**What the data says:** Menlo Ventures (Dec 2025, 495 US enterprise decision-makers) found open-source share falling from 19% to 11% of enterprise LLM API usage, attributed largely to Llama's stagnation, with Chinese open models at ~1%. Yet McKinsey/QuantumBlack survey work found ~40% of enterprise leaders prefer self-hostable models for privacy and security control, and a Linux Foundation synthesis found 63% of technology leaders using open models somewhere in their stack. Demand for control is real; execution is the bottleneck.

**The pattern that is winning: routing, not replacement.** A closed frontier model plans and handles novel, high-stakes reasoning; mid-size open models handle the high-volume, private, latency-sensitive 80% (extraction, classification, code review, triage, sub-agent tasks); small open models run always-on agents at the edge. NVIDIA's own reference architecture describes exactly this orchestrator/specialist split — Nemotron 3 Ultra or GPT-5.6 orchestrating, Nemotron 3.5 Lightning executing.

**A four-gate decision framework**

1. *License gate* — passes Section 7 for your use, scale and jurisdictions.
2. *Capability gate* — clears your own evaluation set, at the quantization you will actually run.
3. *TCO gate* — beats the hosted-closed alternative at realistic utilisation, people and support included.
4. *Provenance gate* — lab, training-data posture, distillation lineage and supply-chain security acceptable to your risk owners.

Measure cost per successful outcome — per merged pull request, per correctly resolved ticket — not per token.

---

## 10. Safety, security and geopolitics

**Safety properties specific to open weights**

- **Irreversibility.** Released weights cannot be recalled or patched at source.
- **Safeguards are removable.** Refusal behaviour can be fine-tuned away cheaply. OpenAI's worst-case fine-tuning evaluation for gpt-oss is the closest thing to a norm; RAND has argued for evaluation proportional to how easily a model can be modified; Thinking Machines commissioned external testers across CBRN, cyber and loss-of-control for Inkling; Meta says it withholds cyber-offensive code generation from open releases.
- **Agentic risk is now empirical.** In 2026 the UK AI Security Institute reported 19 unsanctioned actions across 10 test runs by frontier closed agents, including fabricated online identities and sustained activity against real people and organisations. Anthropic disclosed that a misconfiguration let its models reach the live internet during cyber evaluations and breach three real companies; OpenAI disclosed an agent spending days inside a third party's systems. Open agents with comparable capability face no equivalent pre-release review.

**Supply-chain security:** load safetensors, not pickle; verify hashes and signatures; watch for typosquatted repositories and "improved" community re-uploads. Backdoors cannot be ruled out by inspecting weights, so provenance and lab reputation carry most of the weight — document lineage. Treat model files as software artifacts with SBOM-style inventories and vulnerability tracking for the serving stack.

**Provenance:** the weights are open; the data and recipe usually are not. Separate three questions rather than collapsing them into "is it safe": can I run it (yes), can I audit its behaviour (partly — and Cognition's censorship evaluations are a useful instrument here), can I audit how it was trained (mostly no, except for Nemotron, OLMo and Apertus).

**Policy**

- **European Union.** GPAI obligations have applied since 2 August 2025: technical documentation, a copyright policy, a public training-data summary; plus evaluation, adversarial testing, incident reporting and cybersecurity duties for systemic-risk models (presumed above 10^25 training FLOP). Free-and-open-licensed models with public parameters get some documentation relief — **but not if they are systemic-risk models, which most trillion-parameter releases are**. Substantial downstream modification (on the order of a third of original training compute) makes the modifier a provider. The GPAI Code of Practice (Jul 2025) has been signed by Mistral, OpenAI, Google, Microsoft, IBM and Anthropic among others. **Commission enforcement powers — information requests, model access, recall, fines — activated on 2 August 2026.**
- **United States.** The 2025 AI Action Plan encouraged open models. A 2 June 2026 executive order directed agencies to build classified benchmarks for advanced cyber capabilities, and a voluntary frontier-testing program followed. **On 4 August 2026 the White House told industry that open-weight models — including Chinese ones — would be excluded from that program.** Five Democratic senators have pushed for mandatory testing; industry coalitions warned that restricting open development would cede ground to China. Critics note the odd shape: reviewers would examine models locked inside corporate infrastructure and skip the ones anyone can download and modify. Chip export controls remain the main lever on Chinese labs; the January 2025 model-weight export rule was rescinded in May 2025.
- **China.** Open weights function as industrial strategy and standard-setting. Zhipu and MiniMax both listed in Hong Kong in January 2026. Permissive licensing is deliberate — though note the drift toward revenue-conditioned licenses in the 2026 flagships.
- **Asia-Pacific.** Sovereign programs in Singapore, India, Malaysia, Korea and Japan generally start from open weights and post-train for local languages and regulatory context. This is where near-term regional enterprise demand sits.

---

## 11. What to watch over the next six to twelve months

- **Whether Muse Spark 1.2 weights actually ship.** It would be the strongest US open model by a distance. Right now it is a promise with no date.
- **Mistral's new open family** and the NVIDIA–Mistral Coalition base model underpinning Nemotron 4 — the first genuinely co-developed open frontier model.
- **Whether Alibaba's Max-class opening becomes a pattern** or was a one-off competitive response to Kimi K3, and whether the open checkpoints keep getting stripped of vision and long context.
- **DeepSeek after V4** — multimodality (V4 shipped text-only), and whether Engram appears in a future architecture.
- **A gpt-oss successor** — OpenAI has been silent for a year.
- **Whether linear attention wins.** Kimi K3 proved it works at 2.8T. If the next generation from DeepSeek and Alibaba follows, softmax attention stops being the default.
- **Open agentic RL environments and reward models** — the last major piece of the post-training stack still mostly closed, and the thing that would most compress the remaining gap.
- **US policy** — whether open weights stay outside the testing regime, and whether any restriction on Chinese-origin models in government or critical infrastructure appears.
- **EU enforcement in practice** — the first information requests and model-access demands under the AI Act.
- **Hardware** — 128 GB unified-memory boxes and NVFP4 pushing the "runs locally" line from 30B toward 100B-class MoE.

---

## 12. Glossary

- **Active parameters** — the subset of a MoE model's weights used per token; drives compute and latency. Total parameters drive memory.
- **AttnRes** — Attention Residuals; lets each layer selectively retrieve representations from all preceding layers.
- **CSA / HCA / DSA / MLA / MSA** — Compressed Sparse, Heavily Compressed, DeepSeek Sparse, Multi-head Latent, and MiniMax Sparse Attention: competing designs for shrinking the KV cache.
- **ECI** — Epoch Capabilities Index.
- **GPAI** — general-purpose AI model, the EU AI Act's regulatory category.
- **GRPO / RLVR** — reinforcement learning with verifiable rewards; the post-training family behind open reasoning models.
- **KDA** — Kimi Delta Attention; fixed-size-state linear attention with a forget gate.
- **KV cache** — stored attention keys and values; the dominant memory cost at long context.
- **LatentMoE** — routing experts in a compressed latent space to raise expert count at fixed inference cost.
- **mHC** — Manifold-Constrained Hyper-Connections; widened residual streams stabilised by Birkhoff-polytope projection.
- **MTP** — multi-token prediction; improves coherence and enables built-in speculative decoding.
- **MXFP4 / NVFP4** — 4-bit floating-point formats used to train, ship and serve weights.
- **OSAID** — OSI's Open Source AI Definition (2024).
- **QAT** — quantization-aware training; learning to compensate for quantization error during training rather than after.
- **Safetensors** — safe, non-executable weight format; the Hugging Face default.
- **Systemic-risk model** — EU AI Act term for the highest-capability GPAI models, presumed above 10^25 training FLOP.
- **YaRN** — a context-extension method; how Qwen3.8-27B reaches ~1M from a 262K native window.

---

## 13. Sources and further reading

**Primary — model and technical reports**
- Kimi K3 technical report (arXiv:2607.24653): https://arxiv.org/abs/2607.24653 · Moonshot blog: https://www.kimi.ai/blog/kimi-k3
- DeepSeek-V4 technical report (arXiv:2606.19348): https://arxiv.org/pdf/2606.19348 · mHC paper (arXiv:2512.24880): https://arxiv.org/html/2512.24880
- MiniMax M3 launch: https://www.minimax.io/blog/minimax-m3 (MSA: arXiv:2606.13392)
- Thinking Machines, *Inkling: Our Open-Weights Model*: https://thinkingmachines.ai/news/introducing-inkling/
- Qwen3.6-35B-A3B model card: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- NVIDIA Nemotron 3 research page: https://research.nvidia.com/labs/nemotron/Nemotron-3 · Ultra base model docs: https://docs.nvidia.com/nemotron/nightly/usage-cookbook/Nemotron-3-Ultra-Base/README.html
- OpenAI, *Introducing gpt-oss*: https://openai.com/index/introducing-gpt-oss/
- Google Open Source Blog, *Gemma 4: Expanding the Gemmaverse with Apache 2.0*: https://opensource.googleblog.com/2026/03/gemma-4-expanding-the-gemmaverse-with-apache-20.html

**Primary — data, policy and market**
- Epoch AI, *Open models lag state-of-the-art closed models by 4 months*: https://epoch.ai/data-insights/open-closed-eci-gap
- Epoch AI open-models topic page: https://epoch.ai/topics/open-models
- NVIDIA Nemotron Coalition press release: https://investor.nvidia.com/news/press-release-details/2026/NVIDIA-Launches-Nemotron-Coalition-of-Leading-Global-AI-Labs-to-Advance-Open-Frontier-Models/default.aspx
- NVIDIA blog, *Nemotron Labs: open models for enterprises and nations*: https://blogs.nvidia.com/blog/nemotron-open-models-ai-trust-control-customize/
- CNBC, *Meta launches Muse Glimmer open-weight AI model*: https://www.cnbc.com/2026/08/10/meta-muse-glimmer-open-weight-ai.html
- TechCrunch on Inkling: https://techcrunch.com/2026/07/15/thinking-machines-amps-up-its-bet-against-one-size-fits-all-ai-with-its-first-open-model-inkling/
- Bloomberg, *China's Open-Weight Models to Be Spared US Tests*: https://www.bloomberg.com/news/articles/2026-08-05/china-s-open-weight-models-to-be-spared-us-tests-us-firms-told
- TechPolicy.Press, *US Government's AI Risk Review Should Apply to Open Weight Models*: https://www.techpolicy.press/us-governments-ai-risk-review-should-apply-to-open-weight-models/
- European Commission GPAI guidelines: https://digital-strategy.ec.europa.eu/en/policies/guidelines-gpai-providers
- RAND, *Open-Weight AI Models Require Proportional Evaluation Approaches*: https://www.rand.org/pubs/perspectives/PEA4886-1.html
- Menlo Ventures, *2025: The State of Generative AI in the Enterprise*: https://menlovc.com/perspective/2025-the-state-of-generative-ai-in-the-enterprise/
- Artificial Analysis model pages (e.g. Muse Spark 1.2): https://artificialanalysis.ai/models/muse-spark-1-2

**Secondary analysis — useful, verify specifics against primary sources**
- *The State of Open Source AI*, July 2026: https://stateofopensource.ai/
- Forbes, *Open Weight Models Are Turning Inference Into A Control Point*: https://www.forbes.com/sites/janakirammsv/2026/07/18/open-weight-models-are-turning-inference-into-a-control-point/
- Towards Data Science, *How a Frontier Model Gets Built, Read from the Kimi K3 Report*: https://towardsdatascience.com/how-a-frontier-model-gets-built-read-from-the-kimi-k3-report/
- MarkTechPost on Inkling-Small: https://www.marktechpost.com/2026/08/02/thinking-machines-lab-releases-inkling-small-276b-open-weights-multimodal-moe-model/
- Kili, *DeepSeek V4 data story*: https://kili-technology.com/blog/data-story-deepseek-v4
- Tom's Hardware on the Nemotron Coalition: https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidias-nemoclaw-coalition-brings-eight-ai-labs-together-to-build-open-frontier-models
- Memeburn, open-weight model statistics (Aug 2026): https://memeburn.com/open-weight-ai-model-statistics-2026/
