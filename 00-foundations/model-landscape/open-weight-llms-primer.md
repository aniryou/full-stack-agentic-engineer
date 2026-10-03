# Open-Weight LLMs: A Primer

**State of play as of 29 August 2026.** This field changes in weeks. The specs, dates and benchmark figures in this primer come from the best public sources on that date. The primer uses primary sources where they exist, and it marks vendor claims as vendor claims. Make sure that a number is correct before you put it in a design document or a sizing decision.

---

## 1. The short version

- **"Open weight" means the trained parameters are downloadable and runnable on your own hardware.** It does not mean that the training data, the code or the recipe are public. Only a small number of capable models are open in that more complete sense.
- **Chinese labs still set the open frontier**: Moonshot (Kimi K3), DeepSeek (V4), Alibaba (Qwen 3.8), Z.ai (GLM-5.2) and MiniMax (M3). But 2026 brought a real return of Western labs to open weights. Google moved Gemma 4 to Apache 2.0, and NVIDIA released Nemotron 3 Ultra with data and recipes. Thinking Machines released Inkling, and Meta came back with Muse Glimmer. Mistral has a new open family in early access.
- **The capability gap is roughly one release cycle.** Epoch AI measures approximately four months, or ~8 points on its capability index, for 2026 to date. On Artificial Analysis, the top closed model (Claude Opus 5, 63) is six points ahead of the top open model (Kimi K3, ~57).
- **The gap is jagged, not uniform.** Open models are equal or almost equal to closed models on agentic coding and computer use. They are behind on hard knowledge reasoning (HLE, GPQA, ARC-AGI).
- **August 2026 broke two patterns.** Alibaba opened a Max-class flagship for the first time (Qwen3.8-2.4T-A95B). Meta made a commitment to open Muse Spark 1.2. The pattern "closed frontier tier, open tier one generation behind" is now not the only pattern.
- **The most consequential tier for practitioners is 27–35B dense.** Qwen3.8-27B gets 52 on Artificial Analysis, in a checkpoint that runs on one 24 GB GPU.
- **Enterprise adoption is paradoxical.** Menlo Ventures measured the share of open-source models in enterprise LLM usage in 2025. That share decreased from 19% to 11%, although the capability gap became smaller at the same time. The constraint is governance, TCO and operations, not model quality.
- **Licenses become more different again.** Apache 2.0 and MIT are still the most common licenses. But several 2026 flagships have custom licenses with revenue thresholds (Qwen3.8-Max, MiniMax M3, Kimi K3). Read the actual license file.
- **Policy has split.** The enforcement powers of the EU AI Act became active on 2 August 2026. On 4 August 2026, the US excluded open-weight models from its voluntary frontier test program.

---

## 2. What "open weight" actually means

Openness is a spectrum. These are the useful distinctions:

| Tier | What the lab releases | Typical license | Examples |
|---|---|---|---|
| Closed / API-only | Nothing that you can download | Terms of service | Claude Opus 5, GPT-5.6 Sol, Gemini 3.x, Grok 4.6, Muse Spark 1.2 (for now), Mistral Medium 3.5 |
| Open weights, conditional | Weights under a custom license with scale, revenue or field-of-use conditions | Qwen3.8-Max License, MiniMax Community License, Modified MIT, Llama Community License | Qwen3.8-2.4T-A95B, MiniMax M3, Kimi K3, Llama 4 |
| Open weights, permissive | Weights under an OSI-approved license | Apache 2.0, MIT | DeepSeek V4, GLM-5.2, Qwen3.8-27B, Gemma 4, gpt-oss, Mistral Large 3, Inkling, Muse Glimmer |
| Open source (per OSI's definition) | Weights, code and sufficient data information to make almost the same model again | Permissive license and data disclosure | Few frontier-class models meet this definition |
| Fully open | Weights, data, code, checkpoints and recipe | Apache 2.0 or a similar license | AI2 OLMo 3, NVIDIA Nemotron 3, Swiss Apertus, HF SmolLM |

These points are important in practice:

- **OSI's Open Source AI Definition (Oct 2024)** says that an open-source AI system must have data information, code and parameters. Most "open" LLMs fail on data. "Open weight" is the honest term.
- **What even open labs usually keep back:** the pretraining corpus, the RL environments and reward models, and the full post-training recipe. Sometimes the labs also keep back a capability that exists in the closed version of the model. Alibaba's open 2.4T checkpoint is text-only, and its thinking is always on. But the hosted `qwen3.8-max` API adds vision, video and a default 1M context. Meta said that it will keep some capabilities out of its open releases, in particular cyber-offensive code generation.
- **Transparency decreases while capability increases.** Stanford's Foundation Model Transparency Index decreased from 58 to 40. The labs disclose less about data, parameter counts and compute.
- **Variants per family:** base, instruct, thinking or reasoning (usually with an effort dial), coder, vision, omni, guard/safety, embedding. Distilled small models are a large share of the ecosystem.

---

## 3. How we got here

| When | What | Why it mattered |
|---|---|---|
| 2019 | Staged GPT-2 release | Set the terms of the debate about release norms |
| 2022 | BLOOM, OPT, Pythia | Research-grade open models |
| Feb 2023 | LLaMA leaks | llama.cpp, Alpaca and LoRA appear, and the ecosystem forms |
| Jul–Dec 2023 | Llama 2, Mistral 7B, Mixtral 8x7B | Open weights that you can use commercially. MoE becomes common |
| 2024 | Llama 3.1-405B, Qwen 2.5, Gemma, DeepSeek-V2 (MLA) and V3 (FP8, ~$5.6M run). OSI publishes OSAID | 405B matched the closed frontier for a short time. Chinese labs take the lead in efficiency |
| Jan 2025 | DeepSeek-R1 under MIT | An open release reproduces RL reasoning. The market has a shock |
| 2025 | Qwen3, Kimi K2 (1T, Muon), GLM-4.5–4.7, gpt-oss (Aug). Llama 4 disappoints and Meta puts Behemoth aside. Mistral Large 3 and Ministral 3 (Dec), OLMo 3, Granite 4, Nemotron 3 Nano | Chinese labs are far ahead in release cadence. Reports say that Meta moves to closed models |
| Feb–Mar 2026 | Qwen 3.5 (397B-A17B), Nemotron Coalition (16 Mar), Mistral Small 4, Leanstral, Forge | Open development by coalitions starts |
| Apr 2026 | Gemma 4 Apache 2.0 (2 Apr), Meta's closed Muse Spark (8 Apr), Qwen 3.6 open tier (22 Apr), DeepSeek V4 (24 Apr) | Google stops the use of its custom license. DeepSeek releases 1.6T/1M under MIT |
| May–Jun 2026 | Qwen 3.7 closed (May), MiniMax M3 (1 Jun), Nemotron 3 Ultra (4 Jun), Kimi K2.7 Code (12 Jun), GLM-5.2 (13 Jun) | Three frontier open code models in two weeks |
| Jul 2026 | Inkling (15 Jul), Kimi K3 (16 Jul, weights 27 Jul), Inkling-Small (30 Jul), Mistral open family in early access, Muse Spark 1.1 (9 Jul) | The largest open release ever. A credible new US entry at the open frontier |
| Aug 2026 | Muse Spark 1.2 and Muse Code (5 Aug). US exempts open weights from tests (4 Aug). EU enforcement powers (2 Aug). Muse Glimmer (10 Aug). Qwen3.8-2.4T-A95B (12 Aug). DeepSeek V4-Pro GA (13 Aug). Qwen3.8-27B (14 Aug) | Alibaba opens a Max-class model. Meta goes back to open weights. Governments draw policy lines |

---

## 4. The families, vendor by vendor

The closed reference points for calibration are Claude Opus 5 (AA Intelligence Index 63) and Claude Fable 5 (62). The others are GPT-5.6 Sol (61), Grok 4.6 (61) and Muse Spark 1.2 (57).

### 4.1 Moonshot AI — Kimi (China)

**Current:** Kimi K3 (announced 16 Jul 2026, weights 27 Jul, arXiv:2607.24653). It has 2.8T total / **104B active** parameters, native vision and a 1M context. Its license is Modified MIT. Kimi K2.7 Code (12 Jun) is still a ~1T code specialist with MoonViT vision.

**Position:** Kimi K3 is the strongest open model on most aggregate indices. It is also the largest open-weight release to date. Moonshot held the open scale frontier for nine of the past twelve months.

**Innovations that matter:**

- **Kimi Delta Attention (KDA)**: a linear attention with a state of constant size and a forget gate, from the *Kimi Linear* line of work. Each block of K3 has three KDA layers and one Gated MLA layer. The low-cost linear mix does most of the sequence work. The periodic full-capacity attention keeps the global interaction. K3 is the first and largest public model built mainly on linear attention. It is the strongest signal yet that softmax attention will not stay universal.
- **Attention Residuals (AttnRes)**: each module can select and retrieve representations from the embedding, its own block and all the blocks before it. It does not only read the layer immediately before it. This makes the information path shorter in deep networks.
- **Stable LatentMoE**: 896 routed experts with 16 active (~56x sparsity, up from K2's 384/8). The shared experts run at the full hidden width. The routed experts operate in a narrower latent space: a down-projection occurs before the dispatch, and an up-projection occurs after the aggregation. Three things make the design stable: the SiTU-GLU activation, RMSNorm on the routed experts and Quantile Balancing.
- **Per-head Muon** is the optimizer. K3 also has **quantization-aware training from the SFT stage** (not post-hoc), and Moonshot releases the checkpoint in MXFP4. Thus the quantized checkpoint loses much less quality than a post-training quant does.
- The net effect is approximately **2.5x the scaling efficiency of K2**. Moonshot converts compute into capability, and does not only add parameters.

**Deployment reality:** the model is ~594 GB quantized and ~1.56 TB at BF16. It runs only at cluster scale. Most teams will use it through a hosted endpoint.

**License watch:** Modified MIT. Attribution obligations start above approximately 100M monthly users or $20M monthly revenue.

### 4.2 DeepSeek (China)

**Current:** V4-Pro (1.6T / ~49B active) and V4-Flash (284B / ~13B active). DeepSeek released both on 24 Apr 2026 under **MIT**. Both have 1M context with 384K max output, and both are **text-only**. V4-Pro reached GA on 13 Aug 2026 (checkpoint V4-Pro-0813). The technical report is arXiv:2606.19348.

**Position:** DeepSeek is the price-performance and efficiency benchmark for the whole field. Reports put V4-Pro-Max at 80.6% SWE-bench Verified, the top open-weight entry. They also give it a Codeforces rating of 3206, which makes it the first open model that equals a closed system in competitive code contests.

**Innovations that matter:**

- **Manifold-Constrained Hyper-Connections (mHC)** (arXiv:2512.24880, co-authored by CEO Liang Wenfeng). Hyper-connections make the residual stream wider, with multiple parallel streams. But if the matrices that mix the streams have no constraint, they destroy the training stability. DeepSeek measured signal amplification above 3000x and catastrophic divergence at 27B. The mHC design uses Sinkhorn–Knopp to project those matrices onto the Birkhoff polytope, and this holds the amplification to ~1.6x for ~6.7% training overhead. The mHC design is the single most-cited architectural idea from the open ecosystem this year.
- **Hybrid attention: Compressed Sparse Attention (CSA) + Heavily Compressed Attention (HCA)**. It replaces the Multi-head Latent Attention of V3.2. At 1M tokens, V4-Pro uses approximately **27% of the single-token inference FLOPs and 10% of the KV cache** of V3.2. That is the number that drives the pricing.
- **Muon optimizer**. DeepSeek used 32T training tokens. It trained separate domain specialists and then merged them.
- **Engram** (conditional memory through scalable lookup): many people expected it in V4, but it is **not** in the final architecture. This is a useful correction to much of the early coverage.
- Open kernels: DeepGEMM / MegaMoE. They give a fine-grained overlap of communication and computation in expert-parallel work.

**Economics:** V4-Pro costs ~$0.435 in / $0.87 out per million, and V4-Flash costs ~$0.14 / $0.28. That is approximately 30x below closed-frontier output pricing. V4-Flash is the practical self-host target. V4-Pro fits a single 8-GPU B200 node. But to serve it at competitive latency, you must have cluster economics.

**Operational note:** the legacy `deepseek-chat` and `deepseek-reasoner` aliases went out of service on 24 Jul 2026. Also, most serverless hosts quantize the V4 activations to FP8. This moves the outputs away from the reference weights. If reproducibility is important to you, pin the precision of your host.

### 4.3 Alibaba — Qwen (China)

**Current:** Qwen has the broadest ladder in the ecosystem. It is also the family that changed its posture in August.

- **Qwen3.8-Max** (3 Aug 2026): a 2.4T total / ~95B active sparse MoE with 1M context. It takes text, image and video. It costs $2 / $6 per million on the hosted API.
- **Qwen3.8-2.4T-A95B** (12 Aug 2026): the open-weight Max checkpoint. It is **the first Max-class Qwen ever made downloadable**. It has 512 experts (10 routed + 1 shared). It is a Qwen3.5-family hybrid (Gated DeltaNet + MoE + Gated Attention). Its license is the custom **Qwen3.8-Max License**, not Apache.
- **Qwen3.8-27B** (14 Aug 2026): a 27.78B dense vision-language model under **Apache 2.0**. It has 64 layers and a hidden dim of 5120. Its hybrid layout has 48 Gated DeltaNet linear-attention layers and 16 full-attention layers. It also has multi-token prediction and a `reasoning_effort` dial (xhigh default / medium / low). It has a native 262,144-token context, and YaRN can extend it to ~1M. It takes text, image and video input.
- These are still current: Qwen3.6-27B and Qwen3.6-35B-A3B, Qwen 3.5-397B-A17B, and the full 0.6B–14B ladder.

**Position and the reversal:** until the end of the 3.7 generation, Alibaba kept the frontier closed (3.7-Max, 3.7-Plus, a robotics VLA). It kept the open tier one generation behind. The 3.8 generation reversed that within ten days of launch. But an asymmetry is still there. The open 2.4T checkpoint is **text-only, with thinking forced on and native 262K context**. The hosted API keeps vision, video, optional thinking and a default 1M window.

**Why Qwen3.8-27B is the important one:** it gets **52 on the Artificial Analysis Intelligence Index**. That is **up 14 points over Qwen3.6-27B on an architecturally identical model**. All of that delta comes from post-training. This post-training result shows most clearly this year that the headroom that still exists is in RL and data recipes, not in parameter count. It runs on ~17 GB at 4-bit (an 18 GB Ollama download) on a single 3090 or 4090. Community members who adjusted the setup report ~114 tok/s for one user on a power-limited 3090, and ~1,000 tok/s aggregate across 64 parallel streams.

**Benchmarks, with the asterisk:** Alibaba's own card reports SWE-bench Pro 61.7, Terminal-Bench 2.1 73.0 (up from 63.4) and OSWorld-Verified 84.3 (up from 63.9). It also reports LiveCodeBench v6 90.3 and IFBench 79.5. On the card, the model beats Claude Opus 4.6 Max on 16 of 24 rows. It loses on GPQA Diamond, Terminal-Bench and Humanity's Last Exam (30.8 against 40.0). Several of those benchmarks are in-house or "corrected" versions.

Independent results are now available. Qwen3.8-27B is #1 open-weight on Harvey's Legal Agent benchmark and #9 overall on Code Arena's WebDev leaderboard. It is also #1 open-weight (#7 overall) on Arena.ai's Image-to-WebDev board.

**Known quirk:** at the default `xhigh` reasoning effort, the model thinks far too much about simple prompts. Select the level of the dial yourself.

**License watch:** Qwen3.8-27B is clean Apache 2.0. The 2.4T checkpoint is not. Its license is an MIT-style grant with an attribution rider above 100M MAU or $20M monthly revenue. It also has a **separate paid license for model-as-a-service or AI-work-assistant businesses above $50M trailing-twelve-month revenue, affiliates included**. The paid license does not apply to use that is only internal. Send that paid-license clause to your legal team.

### 4.4 Z.ai / Zhipu — GLM (China)

**Current:** GLM-5.2 (13 Jun 2026), a ~744–753B MoE under **MIT**. It is text-only, with 1M context and explicit reasoning-effort presets.

**Position:** GLM-5.2 is the pragmatic default when you host a model yourself. It does not have the top score. But it has an unambiguous MIT license, the most mature quantization tools of the Chinese flagships, and heavy routing traffic in practice. This combination makes it the model that teams actually run. One gateway reported that GLM-5.2 carried ~4.5x the traffic of Muse Spark 1.2 over the same week, although Muse Spark 1.2 has a higher score.

**What changed from 5.1:** many people judged GLM-5.1 architecturally weak. In a single generation, 5.2 moved to Tier A on independent code benchmarks. It has large gains on code tasks, on agentic tasks and on long-horizon persistence. It is also an outlier in math and reasoning, relative to its position on the overall index. Nous Research added it to Hermes Agent within days of release.

**Numbers:** SWE-bench Pro 62.1, Terminal-Bench 2.1 81.0 (vendor). Artificial Analysis v4.1.1 gives it 53. For a short time, that was the highest open-weights score on that board. On the LLM Stats open leaderboard, it has 46.8 overall. On hosted endpoints, it costs approximately $1.40 / $4.40 per million.

**Corporate context:** Zhipu was the first Chinese AI lab to list publicly, in Hong Kong in January 2026. Since then, its market cap increased to record levels.

### 4.5 MiniMax (China)

**Current:** MiniMax M3 (1 Jun 2026), 428B total / ~23B active. It has up to 1M context (512K guaranteed minimum). It is natively multimodal from step 0 of training: text, image and video go in. It can also operate a desktop.

**Position:** M3 is the only open-weight model that has both frontier agentic coding (80.5% SWE-bench Verified, vendor-reported) and native multimodality. For UI automation and screenshot-to-code at low cost, no other open model is equal to it. It costs $0.30 / $1.20 per million at MiniMax's "permanent 50% off" rate.

**Innovation that matters: MiniMax Sparse Attention (MSA)** (arXiv:2606.13392). MSA is a GQA backbone with block-level selection over **real, uncompressed** key-values, not over the compressed latent state that MLA uses. MiniMax argues that this partitions the KV more precisely than DSA or MoBA, and that it does not have MLA's trade-off between compression and precision. The reported effect at 1M context, against M2, is ~15.6x faster decoding and ~9.7x faster prefill. The reported per-token compute at 1M is approximately 1/20th of M2's. MiniMax publishes the MSA kernels separately, under MIT.

**An important license correction:** M3 is **not** MIT. Its license is the custom **MiniMax Community License** (`license: other`, `license_name: minimax-community`). For commercial use, you must show "Built with MiniMax M3" prominently. **An organization that earns over $20M USD annually** from products built on M3 **must get separate prior written authorization from MiniMax**. The permissive MIT license on the MSA kernel repository does not apply to the weights. This is a real compliance gate, not a formality.

### 4.6 Mistral AI (France)

**Current lineup:** Mistral Large 3 (2 Dec 2025) is a 675B total / 41B active MoE under Apache 2.0. It is still the largest Apache-licensed MoE from a Western lab.

Mistral Small 4 (16 Mar 2026) puts reasoning (Magistral), vision (Pixtral) and code (Devstral) into one model. It is a 119B-total, ~6.5B-active MoE under Apache 2.0 (Hugging Face `mistralai/Mistral-Small-4-119B-2603`; verify, 2026-09). "Small" is now the name of the tier, not the footprint. Its FP8 weights are about 120 GB, which needs two H100s or one H200. By contrast, Small 3.x was a 24B dense model that fits one H100.

Ministral 3 comes in 3B / 8B / 14B, all Apache 2.0. The 14B reasoning variant gets 85% on AIME 2025. There are also specialists: Devstral 2 (code), Voxtral (audio/TTS), Leanstral 1.5 (Lean 4 formal proofs), Mistral OCR and Shieldstral. Shieldstral is a 3B open-weights multimodal safety classifier. It accepts plain-language policies at inference time and runs on a single 16 GB GPU. Mistral Medium 3.5 is closed.

**What comes next:** Mensch confirmed a new open-weight family, in **early access since July 2026**. The family is described as a "fat but sparse" MoE. A broader release is expected. The parameter count, the benchmarks and the license terms are still not public. Separately, the **first Nemotron Coalition model is a base model co-developed by Mistral and NVIDIA on DGX Cloud**. **The plan is to release it as open source on completion, and to make it the foundation of Nemotron 4.**

**Position:** Mistral is the European open-weight option. It is the only European frontier lab that releases open weights at scale. It is also a signatory to the EU GPAI Code of Practice. This is important when a deployment is subject to EU rules, or when you must host the weights in a specific jurisdiction.

**Honest read on capability:** Large 3 is a strong non-reasoning model (MMLU-Pro ~73.1, MATH-500 ~93.6 on independent evaluation). But its scores are much lower on reasoning-heavy benchmarks (~40% AIME 2025, ~44% GPQA Diamond). It is also slow for its class, at ~38 tok/s. The summer release will decide if Mistral is at the open frontier or one tier below it.

**License watch:** most releases are Apache 2.0, but not all. A few releases (some audio models and older research models) use CC-BY-NC or the Mistral Research License. Examine the license of each model.

### 4.7 Google DeepMind — Gemma (US)

**Current:** Gemma 4 (2 Apr 2026) comes in five sizes. The sizes are E2B (2.3B effective), E4B (4.5B effective), a 12B unified multimodal, a 26B MoE with ~4B active, and a 31B dense. Google built it on Gemini 3 research. The small variants have 128K context, and the larger variants have 256K. All variants have vision, and E2B and E4B also have audio. It covers 140+ languages.

**The headline is the license, not the benchmarks.** Gemma 4 is the first Gemmaverse release under the **OSI-approved Apache 2.0** license. It replaces the custom Gemma Terms of Use, with their updatable prohibited-use policy and flow-down obligations. There is no compete clause: you can build a product that competes directly with Google's own. If you evaluated an earlier Gemma and rejected it for legal reasons, examine this version again.

**Capability:** at launch, the 31B dense was #3 and the 26B MoE was #6 among open models on the Arena text leaderboard. The 31B had ~1452 Elo. On intelligence-per-parameter, it did better than models up to 20x its size.

**Strategy:** Gemma is a funnel, not charity. If the most capable open models carry Google's DNA, Google Cloud becomes the natural destination when you scale. Gemma is also the base of sovereign deployments. Examples are the automation of state license processes in Ukraine and Project Navarasa across India's 22 official languages.

### 4.8 Meta (US)

**The arc:** Llama 4 (Apr 2025) did not meet expectations, and Meta put Behemoth aside. Through late 2025, many reports said that Meta was about to stop its open-weight releases and move to a closed frontier model, code name Avocado. Meta Superintelligence Labs, under Alexandr Wang, released that model as **Muse Spark** (8 Apr 2026, closed). Then it released **Muse Spark 1.1** (9 Jul), and then **Muse Spark 1.2** (5 Aug) together with **Muse Code**. Muse Code is a co-trained agent that writes code in a terminal.

Muse Spark 1.2 gets **57 on Artificial Analysis**. In index v4.1.1, its score went up from 54. This was the largest single increase in that update. Muse Spark 1.2 has a 1M context at $1.25 / $4.25 per million.

**The return to open:** on 10 August, Meta released **Muse Glimmer**. It is a ~30B dense multimodal model under **Apache 2.0**, distilled from Muse Spark 1.2. Meta built it for always-on local agent workloads. It runs offline on a single 24 GB consumer GPU, at under 20 GB quantized. Zuckerberg announced four things with it:

- a 6,500-word essay ("The Future is for Everyone")
- a $1B community fund for data-centre neighbours
- a call for lower US barriers on open-source AI
- a commitment to **open the weights of Muse Spark 1.2 itself**

**Status as of late August: Muse Spark 1.2 weights have not shipped.** On 10 August, Wang said "coming soon". There is no repository, no license file and no date. If the release occurs, Muse Spark 1.2 will be the strongest US open-weight model by a large margin. It will also be a real rival to the Chinese tier. Until then, it is a promise.

**Also know this:** there is **no Llama 5**. Meta's model site now shows Muse first and lists Llama 4 after it. Third-party forecasts move Llama 5 to 2027. Llama 4 Maverick is still the last Llama flagship, and much of the "Llama 5 spec sheet" content in search results is fiction. Meta's contributor API tier ($0.10 / $0.20) gives the discount in exchange for permission to train on your prompts and completions. Read those terms before you send private code through it.

### 4.9 NVIDIA — Nemotron (US)

**Current:** the Nemotron 3 family: Nano (~31.6B total / 3.2B active), Super (~120B / 12B active) and **Ultra (4 Jun 2026, ~550B total / up to ~55B active)**. There is also Nemotron 3.5 Lightning (Aug 2026) for high-volume always-on agents. NVIDIA also has specialist models for robotics, AVs, drug discovery and voice.

**Position:** Nemotron is the most truly open of the large-lab releases. NVIDIA publishes **weights, training data and recipes**. Its technical reports are sufficient to build the models again. This is the nearest thing to OSI-style openness at this scale. For a regulated deployment, that is an important difference in governance.

**Innovations that matter:**

- **Hybrid Mamba-2 + Transformer MoE.** Mamba-2 state-space layers give linear-time complexity over the sequence length. Thus a 1M-token context is economical for long-running agents, not only possible.
- **LatentMoE**: it compresses the tokens into a low-rank latent space before the routing. This permits approximately 4x as many expert specialists at the same inference cost.
- **Multi-Token Prediction**: the model predicts several future tokens in each forward pass. This improves the coherence of the chain of thought. It also gives built-in speculative decoding at serve time.
- **NVFP4 four-bit pretraining** on Blackwell. NVIDIA claims ~5x throughput efficiency and ~30% cost reduction against the best open alternatives.

**Ecosystem proof points** (from NVIDIA reports, but specific and with a named source):

- LangChain adjusted its Deep Agents harness for Nemotron 3 Ultra. It changed only the prompts, tools and middleware, and it did not train the model again. It reached the top agent accuracy among open models, at ~10x lower cost per run than the top closed alternatives.
- Arcee AI post-trained on Blackwell to ~$0.90 per million output tokens. That cost is ~20x less than the cost of comparable closed frontier models. The model is second on PinchBench, and it stays fully open weight.
- Harvey post-trained Ultra on its legal benchmark. It matched the top closed models at ≥10x lower cost per run.
- YTL AI Labs post-trained a Nemotron for Malay.
- Artificial Analysis measured Nemotron 3 Nano Omni at 323 tok/s. It is the fastest model on the BenchLM board of the models that clear its evidence thresholds.

**Nemotron Coalition** (announced 16 Mar 2026): Black Forest Labs, Cursor, LangChain, Mistral AI, Perplexity, Reflection AI, Sarvam and Thinking Machines Lab. These labs develop open frontier models together on DGX Cloud. The first deliverable is the Mistral–NVIDIA base model that will be the foundation of Nemotron 4. The contributions cover multimodal (Black Forest), benchmarks of code tasks in practice (Cursor) and agentic/long-horizon evaluation (LangChain).

### 4.10 Thinking Machines Lab (US)

**Current:** **Inkling** (15 Jul 2026) is a 975B total / ~41B active MoE under **Apache 2.0**. The lab trained it on 45 trillion tokens of text, image, audio and video, and it reasons natively across all four. It has 1M context and a thinking effort that you can control. The lab built it in nine months on NVIDIA GB300 NVL72 systems. **Inkling-Small** (30 Jul) is 276B total / 12B active, also under Apache 2.0.

**Why it matters:** Inkling is the most credible US open-weight entry of 2026. It is also the clearest statement of an alternative business model. Mira Murati's lab explicitly states that Inkling is "not the strongest overall model available today, open or closed". The lab gets no money from the model at all. Its revenue comes from **Tinker**, its fine-tuning platform. The lab takes a risk: it expects that "good enough + fully customizable + free" beats "smartest but locked up."

**The notable result:** Inkling-Small **beats its own parent** on reasoning and agentic rows:

- HLE 31.6 against 29.7
- SWE-bench Verified 80.2 against 77.6
- Terminal-Bench 2.1 64.7
- Toolathlon Verified 54.4 against 45.5
- ARC-AGI-2 40.1 against 36.5
- GPQA Diamond 89.5
- AIME 2026 95.5

But it loses badly on factuality (SimpleQA Verified 20.6 against 43.9, AA-Omniscience −9.0 against 2.1). The lab distilled it from an Inkling checkpoint, and then gave it two more weeks of RL on agentic-coding tasks. That is a clean, public demonstration of the current frontier trade: **RL on a smaller student gets agentic capability, but it costs you world knowledge.** Plan your routing for this trade.

**Safety posture, unusually explicit for an open release:** the lab trained Inkling to an internal safety spec across modalities, with commissioned external testers. It did internal and external evaluations for CBRN, cyber and loss-of-control. It gave attention to sycophancy, vulnerable users and manipulation. The lab claims the strongest built-in safeguards of any open-weights model that it compared on FORTRESS. It also trained Inkling to answer directly on censorship-prone topics. Cognition's Propaganda and Censorship Eval found strong censorship non-compliance, a deliberate contrast with Chinese-origin models.

**Deployment:** at BF16, Inkling-Small must have ≥600 GB aggregated VRAM (4x B300 or 8x H200). The NVFP4 checkpoint decreases the floor to 180 GB. For Inkling itself, nearly 2 TB is necessary. The models are available through Tinker, Databricks, Together, Fireworks, Modal, Baseten, and a free rate-limited OpenRouter endpoint.

### 4.11 OpenAI (US)

**Current:** gpt-oss-120b (116.8B total / 5.1B active) and gpt-oss-20b (21B / 3.6B). OpenAI released them in Aug 2025 under **Apache 2.0**, and it also released gpt-oss-safeguard (Oct 2025). They have 128K context, low/medium/high reasoning effort and attention sinks. OpenAI released them natively **MXFP4-quantized**. Thus the 120b fits one 80 GB GPU, and the 20b runs in ~16 GB.

**Position:** gpt-oss became infrastructure, not a headline. MLCommons added gpt-oss-120b as a standard **MLPerf Inference v6.0 benchmark in March 2026**. The hardware industry now treats it as a reference workload. It is still a common default for self-hosted agent stacks with zero budget.

**The notable absence:** a year later, there is no gpt-oss-2 and no announced successor. OpenAI's open contribution now looks like a one-off strategic release, and also a durable contribution to safety methodology. That contribution is the **worst-case fine-tuning evaluation**: you post-train your own model adversarially, to see what a malicious actor can extract. This evaluation is the nearest thing that the field has to a norm for open releases, and that norm is still in development.

### 4.12 The rest of the field, briefly

- **AI2 OLMo 3** (7B / 32B, with Think variants, Apache 2.0): the reference fully-open model. AI2 publishes all of these: the data (Dolma 3), the code, the intermediate checkpoints and the recipe. It is the correct choice when reproducibility is the requirement.
- **IBM Granite 4** (small dense and hybrid Mamba-2 MoE, Apache 2.0): the emphasis is on enterprise governance, with an ISO 42001-certified pipeline. It is a sensible default for regulated, low-risk workloads.
- **Microsoft Phi-4** (4B–15B, MIT): small-model specialists for reasoning and multimodal.
- **xAI**: Grok 4.5 / 4.6 are closed and near the frontier (4.6 at AA 61). The stated policy of xAI is to open earlier generations. It did this with Grok 2.5 in Aug 2025.
- **Cohere Command A / A+**: strong enterprise RAG models, but CC-BY-NC. You cannot use them commercially unless you buy a license.
- **Other Chinese labs:** ByteDance (Seed-OSS), Tencent (Hunyuan), Baidu (ERNIE 4.5, Apache 2.0), Ant Group (Ling/Ring), Meituan (LongCat), Xiaomi (MiMo).
- **Sovereign and regional:** TII Falcon-H1 (UAE), LG EXAONE and Naver HyperCLOVA X (Korea), Sarvam (India), Apertus (Switzerland), AI Singapore's SEA-LION (Southeast Asian languages). Most APAC enterprise demand for open models is actually in this group.
- **Builders on top:** Arcee AI (post-training specialists), Nous Research (Hermes agents), Reflection AI (US open-frontier startup, Coalition member), Liquid AI (LFM2 edge), Hugging Face (SmolLM).
- **Anthropic** releases no open weights.

---

## 5. The innovation map: what actually changed in 2026

Five threads go across every family in §4.

**1. Labs now rebuild attention around the KV cache, not the FLOPs.** At 1M context, the cache is the binding memory constraint, not the weights. Four different approaches are now in production, and each one is an uncertain choice:

- *Compression*: DeepSeek's lineage from MLA to CSA + HCA. It has the lowest cost. It also has a precision cost, which MiniMax explicitly criticises.
- *Block selection on uncompressed KV*: MiniMax's MSA. It keeps the precision, but it has more machinery.
- *Linear attention with periodic full attention*: Kimi's KDA + Gated MLA, and Qwen's Gated DeltaNet + full-attention layers. The recurrent state has a constant size. A 2.8T model now proves the approach.
- *State-space hybrids*: Mamba-2 layers in Nemotron 3, Granite 4 and Falcon-H1. They are linear-time by construction.

There is no convergence yet, and the choice directly sets your serving cost. It also decides if a GGUF fallback keeps the advantage. MiniMax M3 shows this: its sparse attention advantage mostly disappears on the dense-attention local inference path. Most people without a GPU rack will actually take that path.

**2. MoE sparsity increases fast, and stability is the bottleneck.** In one generation, Kimi went from 384 experts / 8 active to 896 / 16 (~56x sparsity). DeepSeek V4-Pro activates ~3% of 1.6T, and Qwen3.8-Max activates ~4% of 2.4T. Each lab that pushed sparsity had to solve activation explosions and expert-utilisation collapse. That is the reason for Stable LatentMoE, Quantile Balancing, SiTU-GLU (Moonshot), LatentMoE with low-rank routing (NVIDIA) and shared-plus-fine-grained expert designs in general.

**3. The residual stream itself is now a design surface.** DeepSeek's mHC makes it wider under a manifold constraint, and Kimi's AttnRes lets layers reach back across depth. Both attack the same problem: in deep, wide models, one narrow residual path is a bottleneck for the information flow. Labs validated both ideas at scale in the same quarter. Expect more work in this area.

**4. Quantization moved from post-hoc to native.** OpenAI released gpt-oss in MXFP4, and Nemotron 3 pretrains in NVFP4. Kimi K3 does quantization-aware training from the SFT stage. "Full precision" and "quantized" become more similar. This quietly removes one of the standard objections to a self-hosted deployment.

**5. The headroom that still exists is in post-training.** The cleanest evidence of the year: **Qwen3.8-27B gained 14 Artificial Analysis points over Qwen3.6-27B on an architecturally identical model**. The cost was approximately 2x the tokens per task. Inkling-Small beat its own 3.5x-larger teacher on agentic rows after two more weeks of RL on agentic-coding tasks.

The ladder is now standard: SFT, then RL with verifiable rewards (the GRPO lineage from DeepSeek-R1), then agentic RL in tool and computer-use environments. Users see this ladder as reasoning-effort dials and thinking-context preservation. Open **agentic RL environments and reward models are still the last major part of the stack that is mostly closed.**

---

## 6. Where the frontier is now

**The closed frontier (late August 2026), on Artificial Analysis Intelligence Index:**

| Model | Lab | AA Index | Notes |
|---|---|---|---|
| Claude Opus 5 (24 Jul) | Anthropic | 63 | It also leads the Agentic Index at 55.3 and ARC-AGI-3 at 30.16%. $5 / $25 |
| Claude Fable 5 | Anthropic | 62 | #1 Arena text (1508.6), HLE 53.3%, AA-Omniscience 40, ARC-AGI-1 leader |
| GPT-5.6 Sol | OpenAI | 61 | ARC-AGI-2 92.5% verified, Terminal-Bench 2.1 88.8% |
| Grok 4.6 | xAI | 61 | Back in the frontier group |
| Muse Spark 1.2 | Meta | 57 | Weights promised, not released |
| Gemini 3.7 Flash | Google | — | The price-performance outlier: ARC-AGI-2 84.6% at ~$0.25/task |

**The open frontier:**

| Model | Lab | Size | License | Standing |
|---|---|---|---|---|
| Kimi K3 | Moonshot | 2.8T / 104B | Modified MIT | #1 open. AA ~57. LLM Stats open leaderboard 55.4. At one point #3 overall on AA, behind only Fable 5 and GPT-5.6 Sol. #1 on Frontend Code Arena. 3rd on LM Arena Agent (0.142 against Opus 5's 0.176) |
| Qwen3.8-2.4T-A95B | Alibaba | 2.4T / 95B | Custom | Newest Max-class open checkpoint. Text-only, but the API is not |
| DeepSeek V4-Pro | DeepSeek | 1.6T / 49B | MIT | 80.6% SWE-bench Verified (V4-Pro-Max), Codeforces 3206. Best price-performance in the field |
| Inkling | Thinking Machines | 975B / 41B | Apache 2.0 | 77.6% SWE-bench Verified. Strongest US open entry. Explicitly does not try to be #1 |
| GLM-5.2 | Z.ai | ~750B | MIT | AA 53. The practical self-host default. Highest actual routing traffic |
| MiniMax M3 | MiniMax | 428B / 23B | Community | 80.5% SWE-bench Verified. Only open model with native multimodality, 1M and frontier agentic coding |
| Inkling-Small | Thinking Machines | 276B / 12B | Apache 2.0 | 80.2% SWE-bench Verified. Beats its own parent on agentic rows |
| Qwen3.8-27B | Alibaba | 27.8B dense | Apache 2.0 | AA 52 on a single 24 GB GPU. The best intelligence-per-gigabyte available |

**How wide is the gap, measured three ways:**

- **Epoch AI:** since January 2026, the best open models are behind the closed frontier. The average gap is **four months, or ~8 ECI points** (90% CI 7–11). That is wider than the three-month average measured over 2023–2025. Under a stricter comparison rule, the gap is ~six months. Epoch's July index put Kimi K3 at 156 against GPT-5.6 Sol at 162, with **confidence intervals that overlap**.
- **Artificial Analysis:** Opus 5 at 63 against Kimi K3 at ~57, a gap of six points. Of 170–185 tracked models, approximately 95 are open weight.
- **Stanford AI Index (March 2026 snapshot):** 1,503 Arena points for the top closed model against 1,454 for the top open model. That is a 3.3% gap, against 0.5% in August 2024. The US–China gap at the top is ~2.7%.

**Two reasons the real gap is probably wider than these numbers.** Epoch gives both reasons. First, open models appear to optimize more aggressively for public benchmarks, and they appear to do relatively worse on private evaluations. Second, closed labs keep back their strongest models. Thus the comparison puts open models against the *published* frontier, not the actual one.

**The shape of the frontier, honestly:**

- **Open is at parity or ahead** on agentic coding (multiple open models above 80% SWE-bench Verified). It is also at parity or ahead on computer use and desktop control. Reports put Qwen3.8-Max at 86.1 OSWorld-Verified, ahead of GPT-5.6 Sol Max and Fable 5. Open is also at parity or ahead on instruction following, long-context retrieval, extraction, classification and most multilingual work.
- **Open is behind** on hard knowledge reasoning and on the solution of abstract problems. Examples are Humanity's Last Exam (Qwen3.8-27B 30.8 against Opus 4.6 Max 40.0, Fable 5 at 53.3), GPQA Diamond and ARC-AGI-2/3. Open is also behind on research synthesis, and on reliability for truly new tasks.
- **Open has won on price and forced the closed tier to respond.** Open API list prices are ~8x below closed prices on average, and DeepSeek's are ~30x below on output tokens. On 30 July, OpenAI cut the Terra price by 20% and the Luna price by 80%. It gave improvements in serving cost as the reason. That was the steepest cut of the year from a US lab, and a direct answer to the Chinese open tier.
- **Usage and enterprise adoption still show different results.** Reports put Chinese models at ~61% of OpenRouter traffic in June 2026. Menlo's enterprise survey put Chinese open models at ~1% of LLM API usage in enterprises. Both are true, because the populations are different.

**The three structural shifts to take away:**

1. **Scale has divided into two groups.** The open flagships are 0.4–2.8T MoE models that almost nobody hosts on their own hardware. People use them through hosted endpoints. There, the open/closed difference becomes only a question of licenses and provenance, not of deployment. The models that people actually run are 4–35B. That tier got much better in 2026.
2. **The pattern of tiers starts to break down, and the change is uneven.** Alibaba opened a Max-class model, and Meta made a commitment to open its flagship. NVIDIA and Thinking Machines open everything. Google, Mistral and Alibaba still keep a closed top tier. Treat open weights as a strategic lever that any lab can withdraw, not as a principle. For example, Alibaba released its 3.7 generation entirely closed, one generation before it opened a 2.4T flagship.
3. **America re-entered, but China still sets the pace.** Muse Glimmer, Inkling, Nemotron 3 Ultra and gpt-oss are real US open contributions. Nemotron is the most transparent release from any large lab. But the top of the open leaderboard is Moonshot, DeepSeek, Alibaba, Z.ai and MiniMax. The architectural innovations that everyone now copies are MLA/CSA, mHC, KDA, LatentMoE, MSA and Muon at scale. They came mostly from labs that operate under GPU export constraints.

---

## 7. Licensing: what to actually check

| License | Commercial use | Notable terms | Used by |
|---|---|---|---|
| Apache 2.0 | Yes | Patent grant, attribution, no copyleft | Qwen3.8-27B and open Qwen tier, most Mistral, Gemma 4, gpt-oss, Muse Glimmer, Inkling, Granite, OLMo, ERNIE 4.5 |
| MIT | Yes | Minimal terms, no patent grant | DeepSeek V3/R1/V4, GLM-5.x, Phi |
| Modified MIT (Moonshot) | Yes | Attribution above ~100M MAU or ~$20M monthly revenue | Kimi K2 / K3 |
| Qwen3.8-Max License | Yes, with limits | MIT-style grant and an attribution rider above 100M MAU / $20M monthly revenue. **A separate paid license for MaaS or AI-work-assistant businesses above $50M TTM revenue, affiliates included**. Internal use is outside the paid license | Qwen3.8-2.4T-A95B |
| MiniMax Community License | Yes, with limits | "Built with MiniMax M3" attribution. **Prior written authorization is necessary above $20M annual revenue** | MiniMax M3, M2.7 |
| Open model license with data and recipes | Yes | It covers weights and training data. It is permissive | Nemotron 3 family |
| Llama Community License | Yes, with limits | 700M-MAU threshold, "Built with Llama" name requirement, AUP. Llama 4 restricts EU-domiciled multimodal use | Llama 2–4 |
| Gemma Terms of Use | Yes, with limits | Updatable prohibited-use policy, flow-down | Gemma 1–3 (Gemma 4 is Apache 2.0) |
| CC-BY-NC 4.0 | No | Non-commercial only | Cohere Command A, some Mistral audio releases |
| Mistral Research License | No | Research only. Commercial use through Mistral | Older Mistral models |

**Diligence checklist**

1. **Scale and revenue thresholds**: MAU, monthly revenue, TTM revenue with affiliates included, and what occurs when you cross them. Three of 2026's flagship open models have these.
2. **Field-of-use restrictions**: the Qwen3.8-Max MaaS clause is the clearest example. If you resell inference, read it first.
3. **Acceptable-use policies**: can the provider update them without your agreement? Do they bind your downstream users?
4. **Attribution and name requirements**: "Built with X" in product names or docs.
5. **Outputs**: who owns them, and if you can train other models on them.
6. **Derivatives**: can you release fine-tuned weights, and under what license?
7. **Patent grant**: Apache 2.0 has one. MIT does not.
8. **Jurisdiction and export controls**: geographic exclusions (Llama 4's EU restriction) and your own export exposure.
9. **Data you add**: fine-tuning datasets have their own licenses. Synthetic data from closed APIs can have terms that restrict the training of models that compete with the provider.
10. **Hygiene**: in May 2026, approximately 70% of scanned Hugging Face repositories had no license tag. The absence of a tag is not permission. Examine the LICENSE file on the exact checkpoint. Also, a permissively licensed kernel repository (MiniMax's MSA) does not give a license for the weights.
11. **Indemnity and support**: cloud catalogs (Bedrock, Vertex, Azure AI Foundry) sell indemnified versions. Vendors such as Mistral, IBM and NVIDIA also sell them. That is usually what compliance actually wants.

---

## 8. Running, adapting and operating open models

### Memory math

- **Weights:** total parameters × bytes per parameter (BF16 = 2, FP8 = 1, INT4/FP4 ≈ 0.5–0.6 with scales), plus 10–20% overhead. For MoE, **all** experts must be in memory, although few of them are active.
- **KV cache:** context × layers × KV heads × head dim × bytes. MLA, sparse attention, linear layers and SSM layers make it much smaller. At 1M context, it is the largest part of the memory.
- **Worked examples (approximate):**
    - gpt-oss-20b ~16 GB
    - Qwen3.8-27B ~17 GB at 4-bit (18 GB Ollama download), ~28 GB FP8, ~56 GB BF16 before KV
    - Muse Glimmer <20 GB quantized
    - Gemma 4 31B ~18–20 GB at 4-bit
    - gpt-oss-120b ~65 GB MXFP4 (one 80 GB GPU)
    - Inkling-Small 180 GB NVFP4 / ≥600 GB BF16
    - DeepSeek V4-Flash ~150–170 GB at 4-bit
    - DeepSeek V4-Pro fits one 8-GPU B200 node at low precision
    - Qwen3.8-2.4T above 400 GB even at aggressive 1-bit quants
    - Kimi K3 ~594 GB quantized, ~1.56 TB BF16, multi-node only

### Deployment tiers

| Tier | Hardware | What runs well |
|---|---|---|
| Laptop / edge | 8–32 GB RAM, NPU or consumer GPU | Gemma 4 E2B/E4B, Ministral 3, Qwen 3.6-4B, Phi |
| Workstation | 24–32 GB GPU, or 128 GB unified memory (DGX Spark, Strix Halo, M-series Max) | **Qwen3.8-27B, Gemma 4 31B, Muse Glimmer, gpt-oss-20b**, and 100B-class MoE on 128 GB |
| Single server | 1–8 × 80–192 GB datacenter GPUs | gpt-oss-120b, V4-Flash, Nemotron Super, Mistral Large 3, Inkling-Small (NVFP4) |
| Cluster | Multi-node H100/H200/B200/GB300 | V4-Pro, GLM-5.2, Kimi K3, Nemotron 3 Ultra, Inkling, Qwen3.8-Max |
| Hosted open endpoints | None of your own | Everything in the other rows. But quantization and version policies are different from host to host |

### The serving stack

- **Engines:** vLLM and SGLang, TensorRT-LLM, and llama.cpp / Ollama / LM Studio / MLX locally. vLLM and SGLang have day-0 support for most major releases. For example, DeepSeek V4 had both on release day.
- **Cost levers:** paged and prefix KV caching, continuous batching, disaggregated prefill/decode, speculative decoding (MTP heads), FP8/FP4 kernels, expert parallelism.
- **Formats:** safetensors for weights (never load pickle from untrusted sources), GGUF for llama.cpp, and AWQ/GPTQ/MXFP4/NVFP4/Q4_K_M quants. Sparse-attention models must use their documented serving flags. A 428B MoE with MSA will not serve well on generic settings.
- **Hosted providers:** OpenRouter, Together, Fireworks, Groq, Cerebras, Baseten, DeepInfra, Lambda, Morph, and also hyperscaler catalogs and regional clouds. Most have OpenAI-compatible APIs, and more and more of them also have Anthropic-compatible APIs. Thus agent harnesses change models through environment variables.

### The adaptation ladder (cheapest first)

1. **Prompt and harness adjustment**: frequently the largest single improvement. LangChain adjusted only the prompts, tools and middleware, and thus reached top open-model agent accuracy on Nemotron 3 Ultra. Also, Meta measures Muse Spark 1.2 inside its co-trained Muse Code harness. In vendor charts, you cannot separate the model from the harness.
2. **Retrieval and tool grounding**: the same as for closed models. But you can put the model in the same location as the data.
3. **PEFT**: LoRA/QLoRA through TRL, Unsloth, Axolotl or LLaMA-Factory. It takes hours on one GPU for a 27B model.
4. **Full post-training and RL**: NeMo, TRL and verl-style stacks, or managed routes (Mistral Forge, Thinking Machines' Tinker). Harvey reached frontier-class legal accuracy on Nemotron at ≥10x lower cost per run. Arcee got to ~$0.90 per million output tokens.
5. **Distillation**: compress a large open teacher into a task-specific student. Inkling-Small is the public worked example, and it also shows the factuality cost. For how it works, and when a student pays for itself, see [`distillation`](../distillation/README.md).

### Operating discipline

- **Governance:** in production, record the model version, the quantization format, the serving engine version and the license text. When three quants and two point releases are in use, "We run GLM-5.2" is not a complete answer. Forrester's Model Openness Framework (Apr 2026) and Gartner's AI TRiSM material are usable as a framework.
- **Safety layers:** use a guard classifier (Shieldstral, gpt-oss-safeguard, Llama Guard) with the model. Also use prompt-injection controls and output controls. Fine-tuning can change built-in refusals. Thus they are not a control that you can depend on alone.
- **Lifecycle:** no provider deprecates the model under you. But you own the upgrades, the regression tests and the patches to the serving stack.
- **Cost reality:** do not host models yourself on rented GPUs unless you keep them saturated. Below that level, a hosted open endpoint costs less. In 2026, the OECD published a break-even model. It is a useful TCO template.

---

## 9. Why enterprises choose open weights — and why they often don't

**Drivers:**

- data residency and sovereignty
- cost at volume
- latency, edge and offline operation
- customisation, where open models most often *beat* closed models for a specific task
- predictability, and independence from silent deprecations
- auditability of a checkpoint that does not change

**Frictions:**

- operational burden and capex
- capability lag on the hardest tasks, and worse behaviour on private benchmarks than the leaderboards suggest
- license diligence, where most deals actually stall
- provenance and security
- support and indemnity, which usually make a vendor necessary and thus remove part of the cost advantage

**What the data says:** Menlo Ventures did a survey (Dec 2025, 495 US enterprise decision-makers). It found that the open-source share of LLM API usage in enterprises decreased from 19% to 11%. The survey says that this is mostly because of Llama's stagnation, and it puts Chinese open models at ~1%. But McKinsey/QuantumBlack survey work found that ~40% of enterprise leaders prefer self-hostable models for privacy and security control. A Linux Foundation synthesis found that 63% of technology leaders use open models somewhere in their stack. The demand for control is real, but execution is the bottleneck.

**The pattern that wins: routing, not replacement.** A closed frontier model plans and does the new, high-stakes reasoning. Mid-size open models do the high-volume, private, latency-sensitive 80% (extraction, classification, code review, triage, sub-agent tasks). Small open models run always-on agents at the edge. NVIDIA's own reference architecture describes exactly this orchestrator/specialist split. In it, Nemotron 3 Ultra or GPT-5.6 is the orchestrator, and Nemotron 3.5 Lightning does the execution.

**A four-gate decision framework**

1. *License gate*: the model passes Section 7 for your use, scale and jurisdictions.
2. *Capability gate*: the model passes your own evaluation set, at the quantization that you will actually run.
3. *TCO gate*: the model beats the hosted-closed alternative at realistic utilisation, with people and support included.
4. *Provenance gate*: your risk owners accept the lab, the training-data posture, the distillation lineage and the supply-chain security.

Measure the cost per successful outcome, not the cost per token. For example, measure the cost per merged pull request or per correctly resolved ticket.

---

## 10. Safety, security and geopolitics

**Safety properties specific to open weights**

- **Irreversibility.** After a lab releases weights, it cannot recall them or patch them at the source.
- **Safeguards are removable.** A low-cost fine-tune can remove refusal behaviour. OpenAI's worst-case fine-tuning evaluation for gpt-oss is the nearest thing to a norm. RAND argued for evaluation that is proportional to how easily someone can change a model. Thinking Machines commissioned external testers across CBRN, cyber and loss-of-control for Inkling. Meta says that it keeps cyber-offensive code generation out of open releases.
- **Agentic risk is now empirical.** In 2026, the UK AI Security Institute reported 19 unsanctioned actions across 10 test runs by frontier closed agents. These included fabricated online identities and sustained activity against real people and organisations. Anthropic disclosed that a misconfiguration let its models reach the live internet during cyber evaluations and breach three real companies. OpenAI disclosed that an agent spent days inside a third party's systems. Open agents with comparable capability have no pre-release review of the same kind.

**Supply-chain security:** load safetensors, not pickle. Verify hashes and signatures. Look carefully for typosquatted repositories and "improved" community re-uploads.

An inspection of the weights cannot exclude backdoors. Thus provenance and lab reputation carry most of the weight. Record the lineage. Treat model files as software artifacts with SBOM-style inventories. Monitor the serving stack for vulnerabilities.

**Provenance:** the weights are open, but the data and the recipe usually are not. Ask three separate questions. Do not combine them into one question, "is it safe":

- Can I run it? Yes.
- Can I audit its behaviour? Partly. Cognition's censorship evaluations are a useful instrument here.
- Can I audit how the lab trained it? Mostly no, except for Nemotron, OLMo and Apertus.

**Policy**

- **European Union.** GPAI obligations started to apply on 2 August 2025: technical documentation, a copyright policy and a public training-data summary. Systemic-risk models (presumed above $10^{25}$ training FLOP) also have duties of evaluation, adversarial tests, incident reports and cybersecurity. Free-and-open-licensed models with public parameters get some documentation relief, **but not if they are systemic-risk models, which most trillion-parameter releases are**.

    If a downstream party makes a substantial change (on the order of a third of original training compute), that party becomes a provider. Mistral, OpenAI, Google, Microsoft, IBM and Anthropic, among others, signed the GPAI Code of Practice (Jul 2025). **The Commission enforcement powers (information requests, model access, recall, fines) became active on 2 August 2026.**

- **United States.** The 2025 AI Action Plan encouraged open models. A 2 June 2026 executive order told agencies to build classified benchmarks for advanced cyber capabilities. A voluntary frontier test program came after it. **On 4 August 2026, the White House told industry that open-weight models, Chinese ones included, will stay outside that program.**

    Five Democratic senators pushed for mandatory tests. Industry coalitions warned that restrictions on open development will give China an advantage. Critics say that the shape of the policy is odd. In the view of the critics, the reviewers will examine models locked inside corporate infrastructure, and skip the models that anyone can download and change. Chip export controls are still the main lever on Chinese labs. The US cancelled the January 2025 model-weight export rule in May 2025.

- **China.** Open weights operate as industrial strategy and as a way to set standards. Zhipu and MiniMax both listed in Hong Kong in January 2026. The permissive licenses are deliberate. But the 2026 flagships show a drift toward revenue-conditioned licenses.
- **Asia-Pacific.** Sovereign programs in Singapore, India, Malaysia, Korea and Japan usually start from open weights. Then they post-train for local languages and regulatory context. This is where the near-term regional enterprise demand is.

---

## 11. What to watch over the next six to twelve months

- **If Muse Spark 1.2 weights actually ship.** If Meta releases them, Muse Spark 1.2 will be the strongest US open model by a large margin. At this time, it is a promise with no date.
- **Mistral's new open family**, and the NVIDIA–Mistral Coalition base model that will be the foundation of Nemotron 4. The Coalition base model is the first open frontier model that labs truly develop together.
- **If Alibaba's release of a Max-class model becomes a pattern**, or was a one-off competitive response to Kimi K3. Also watch if Alibaba continues to remove vision and long context from the open checkpoints.
- **DeepSeek after V4**: multimodality (DeepSeek released V4 text-only), and if Engram appears in a future architecture.
- **A gpt-oss successor**: OpenAI said nothing about it for a year.
- **If linear attention wins.** Kimi K3 proved that it works at 2.8T. If the next generation from DeepSeek and Alibaba also uses it, softmax attention will no longer be the default.
- **Open agentic RL environments and reward models**: the last major part of the post-training stack that is still mostly closed. No other change can decrease the gap that is still there as much as open versions of them.
- **US policy**: if open weights stay outside the test regime, and if any restriction on Chinese-origin models in government or critical infrastructure appears.
- **EU enforcement in practice**: the first information requests and model-access demands under the AI Act.
- **Hardware**: 128 GB unified-memory boxes and NVFP4 push the "runs locally" line from 30B toward 100B-class MoE.

---

## 12. Glossary

- **Active parameters**: the subset of a MoE model's weights that the model uses for each token. They control compute and latency. Total parameters control memory.
- **AttnRes**: Attention Residuals. Each layer can select and retrieve representations from all the layers before it.
- **CSA / HCA / DSA / MLA / MSA**: Compressed Sparse, Heavily Compressed, DeepSeek Sparse, Multi-head Latent, and MiniMax Sparse Attention. These are rival designs to make the KV cache smaller.
- **ECI**: Epoch Capabilities Index.
- **GPAI**: general-purpose AI model, the regulatory category of the EU AI Act.
- **GRPO / RLVR**: reinforcement learning with verifiable rewards. It is the post-training family behind open reasoning models.
- **KDA**: Kimi Delta Attention, a linear attention with a state of constant size and a forget gate.
- **KV cache**: the stored attention keys and values. It is the largest memory cost at long context.
- **LatentMoE**: the routing of experts in a compressed latent space. It increases the expert count at a constant inference cost.
- **mHC**: Manifold-Constrained Hyper-Connections. A Birkhoff-polytope projection makes the widened residual streams stable.
- **MTP**: multi-token prediction. It improves coherence and gives built-in speculative decoding.
- **MXFP4 / NVFP4**: 4-bit floating-point formats. Labs use them to train, release and serve weights.
- **OSAID**: OSI's Open Source AI Definition (2024).
- **QAT**: quantization-aware training. The model learns to compensate for the quantization error during training, not after it.
- **Safetensors**: a safe, non-executable weight format. It is the Hugging Face default.
- **Systemic-risk model**: the EU AI Act term for the highest-capability GPAI models, presumed above $10^{25}$ training FLOP.
- **YaRN**: a context-extension method. Qwen3.8-27B uses it to reach ~1M from a 262K native window.

---

## 13. Sources and further reading

**Primary: model and technical reports**

- Kimi K3 technical report (arXiv:2607.24653): https://arxiv.org/abs/2607.24653 · Moonshot blog: https://www.kimi.ai/blog/kimi-k3
- DeepSeek-V4 technical report (arXiv:2606.19348): https://arxiv.org/pdf/2606.19348 · mHC paper (arXiv:2512.24880): https://arxiv.org/html/2512.24880
- MiniMax M3 launch: https://www.minimax.io/blog/minimax-m3 (MSA: arXiv:2606.13392)
- Thinking Machines, *Inkling: Our Open-Weights Model*: https://thinkingmachines.ai/news/introducing-inkling/
- Qwen3.6-35B-A3B model card: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- NVIDIA Nemotron 3 research page: https://research.nvidia.com/labs/nemotron/Nemotron-3 · Ultra base model docs: https://docs.nvidia.com/nemotron/nightly/usage-cookbook/Nemotron-3-Ultra-Base/README.html
- OpenAI, *Introducing gpt-oss*: https://openai.com/index/introducing-gpt-oss/
- Google Open Source Blog, *Gemma 4: Expanding the Gemmaverse with Apache 2.0*: https://opensource.googleblog.com/2026/03/gemma-4-expanding-the-gemmaverse-with-apache-20.html

**Primary: data, policy and market**

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
- Artificial Analysis model pages (for example, Muse Spark 1.2): https://artificialanalysis.ai/models/muse-spark-1-2

**Secondary analysis: useful, but compare the specifics with primary sources**

- *The State of Open Source AI*, July 2026: https://stateofopensource.ai/
- Forbes, *Open Weight Models Are Turning Inference Into A Control Point*: https://www.forbes.com/sites/janakirammsv/2026/07/18/open-weight-models-are-turning-inference-into-a-control-point/
- Towards Data Science, *How a Frontier Model Gets Built, Read from the Kimi K3 Report*: https://towardsdatascience.com/how-a-frontier-model-gets-built-read-from-the-kimi-k3-report/
- MarkTechPost on Inkling-Small: https://www.marktechpost.com/2026/08/02/thinking-machines-lab-releases-inkling-small-276b-open-weights-multimodal-moe-model/
- Kili, *DeepSeek V4 data story*: https://kili-technology.com/blog/data-story-deepseek-v4
- Tom's Hardware on the Nemotron Coalition: https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidias-nemoclaw-coalition-brings-eight-ai-labs-together-to-build-open-frontier-models
- Memeburn, open-weight model statistics (Aug 2026): https://memeburn.com/open-weight-ai-model-statistics-2026/
