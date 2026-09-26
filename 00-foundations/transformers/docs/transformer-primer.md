# Transformers: A Practitioner's Primer

*For readers who know what a neural network is and have trained a model or two, but couldn't sketch a Transformer from memory or say why it's built the way it is.*

**How to read this.** Sections 1–5 build the architecture outward from one idea. Sections 6–7 cover training and inference, which is where most practitioner intuition lives. Section 8 onward is reference material: numbers, modern variants, hard-won intuitions, and a working implementation. If you read only one part, read Section 3 and the mental model at its end.

---

## 1. The problem: a token's meaning depends on its context

Language, code, protein sequences, audio: all are sequences of discrete symbols whose meaning depends on what surrounds them. "Bank" on its own is ambiguous; "bank" three words after "river" is not. A sequence model's job is to turn each symbol into a vector that captures what it means *here*, given everything around it.

Until 2017 the standard answer was the recurrent network (RNN, LSTM): read tokens one at a time, carrying a hidden state forward. Two problems. Information from far back has to survive many sequential updates, and it degrades, so long-range dependencies were hard to learn. More decisively, step *t* can't begin until step *t−1* finishes, so training can't parallelize across the sequence. GPUs are throughput machines; a model that forces serial computation wastes them. Convolutions parallelize, but each layer only sees a fixed local window.

The Transformer (Vaswani et al., 2017, "Attention Is All You Need") dropped recurrence entirely. Every token looks directly at every other token in one step, and every position is computed in parallel. That is the whole trick. Everything else in the architecture is scaffolding that makes the trick trainable at scale.

## 2. The one idea: attention is a soft lookup

### 2.1 Intuition

A dictionary lookup takes a **query**, compares it against **keys**, and returns the **value** stored under the matching key. Attention is the differentiable version: instead of returning one value, it returns a weighted average of *all* values, weighted by how well the query matches each key.

Every token derives three vectors from its current representation, through three learned linear maps:

- **Query (Q)** — what am I looking for?
- **Key (K)** — what do I contain, for matching purposes?
- **Value (V)** — what do I hand over if someone attends to me?

Take "it" in *"The animal didn't cross the street because it was too tired."* A useful query for "it" would encode something like "pronoun, needs an antecedent." The key for "animal" matches that query well; the key for "street" does not. "it" receives a blend of values weighted heavily toward "animal", and its representation now carries "refers to the animal." Nobody programs this; it emerges from training.

### 2.2 The formula, term by term

For n tokens with query, key, and value matrices Q, K, V (each n × d_k, where d_k is the per-head dimension):

    Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) · V

- **Q Kᵀ** is an n × n matrix of dot products: the raw compatibility of every query with every key. Row *i* says how much token *i* wants each token *j*.
- **/ √d_k** keeps those dot products from growing with dimension. Dot products of random d_k-dimensional vectors have variance proportional to d_k; unscaled, the softmax saturates toward one-hot outputs and gradients vanish. A small detail that matters a lot in practice.
- **softmax**, applied row-wise, turns each row into positive weights summing to 1. This is what makes the lookup *soft*: an average, not a selection.
- **· V** takes the weighted average of value vectors. Token *i*'s output is Σⱼ wᵢⱼ · vⱼ.

### 2.3 A toy calculation

Three tokens, two-dimensional vectors, scaling omitted for readability. Token 3's query is q = [2, 0]. The keys are k₁ = [1, 0], k₂ = [0, 1], k₃ = [0, 0]; the values are v₁ = [1, 0], v₂ = [0, 1], v₃ = [1, 1].

    scores  = [q·k₁, q·k₂, q·k₃] = [2, 0, 0]
    weights = softmax([2, 0, 0]) ≈ [0.79, 0.11, 0.11]
    output  = 0.79·v₁ + 0.11·v₂ + 0.11·v₃ ≈ [0.89, 0.21]

Token 3 ends up mostly holding token 1's value, with a little of everything else mixed in. Two things to notice. The weights are a convex combination, so attention averages values rather than amplifying any one. And the output lives in value space: what a token *receives* is determined by W_V, not by the attended token's raw embedding.

### 2.4 Self-attention and cross-attention

When Q, K, V all come from the same sequence it's **self-attention**: tokens contextualize each other. When Q comes from one sequence and K, V from another (a decoder reading an encoder's output; a language model reading image features), it's **cross-attention**. Same math, different inputs. Decoder-only LLMs use self-attention only.

## 3. Anatomy of a Transformer block

A Transformer is a stack of L identical blocks. Each block has two sub-layers, each wrapped in a normalization and a residual connection:

```
token ids
    │
    ▼
embedding lookup (+ positional information)
    │
    ▼
x : the residual stream, shape (n tokens × d_model)
    │
    ▼
┌── one block, repeated L times ─────────────────────────────┐
│                                                            │
│   x = x + Attention(Norm(x))   ← mixes across positions    │
│   x = x + MLP(Norm(x))         ← computes per position     │
│                                                            │
└────────────────────────────────────────────────────────────┘
    │
    ▼
final Norm ──► unembed (linear) ──► softmax ──► P(next token | context)
```

The matrix flowing through has shape (n tokens × d_model). The width d_model is 768 for GPT-2 small, 4096 for a 7B-class model, 12,288 for GPT-3 175B. Every block reads this matrix and adds an update back into it.

### 3.1 Multi-head attention

One attention operation computes one n × n pattern of relationships. But a token has several kinds of relationship worth tracking at once — syntactic (what's my subject?), positional (what came right before me?), semantic (which earlier word means the same as me?). So instead of one attention with d_model-dimensional Q/K/V, a block runs h **heads** in parallel, each of dimension d_head = d_model / h, concatenates their outputs, and applies a final linear map W_O.

GPT-2 small: 12 heads of 64 dimensions. Llama-2 7B: 32 heads of 128. Head dimension is almost always 64 or 128; what scales with model size is the number of heads.

**Practitioner note.** Heads cost nothing extra. One weight matrix of shape (d_model, 3·d_model) produces Q, K, V for all heads in a single matmul, and a reshape splits them. Attention has 4·d_model² parameters per block (W_Q, W_K, W_V, W_O) regardless of h.

### 3.2 The MLP

After attention, each token's vector passes independently through a two-layer feed-forward network: expand to about 4·d_model (or ~2.7× with modern gated variants), apply a nonlinearity (GELU, or SwiGLU today), project back down to d_model. Same weights at every position; no interaction between positions.

This sub-layer is easy to overlook and holds roughly two-thirds of the parameters. Interpretability work suggests it's where much of the factual knowledge lives: the up-projection asks a large bank of "is pattern X present?" questions, the nonlinearity gates them, and the down-projection writes the consequences back into the token's representation. A useful shorthand: **attention decides where to look; the MLP decides what it means.**

### 3.3 Residual connections and normalization

Each sub-layer's output is *added* to its input rather than replacing it: x ← x + f(x). Two consequences.

First, deep stacks become trainable. Gradients flow straight back through the additions, so a 100-block model avoids the vanishing gradients that limited deep networks before ResNets.

Second — and this is the mental model experienced practitioners actually use — there is a single **residual stream** running through the whole network, and each block reads from it and writes a small increment. The stream is a shared workspace, d_model wide, in which a token's representation accumulates. Early blocks tend to write syntactic and positional features, later blocks more abstract ones, and the final representation is the sum of every block's contribution.

**Normalization** (LayerNorm, or more often now RMSNorm) rescales each token's vector to unit scale before each sub-layer. It keeps activations in a numerically comfortable range and makes training far less sensitive to learning rate. The original paper normalized *after* the residual addition ("post-norm"); nearly everything since GPT-2 normalizes *before* the sub-layer ("pre-norm"), which trains stably at large scale without delicate warmup. This is one of the few architectural changes since 2017 that everyone adopted.

### 3.4 The mental model

A Transformer block does exactly two things:

1. **Attention moves information between positions.** It is the only place tokens interact: it reads from other tokens' residual streams and writes into this one.
2. **The MLP processes information within a position.** It transforms what's in this token's stream without looking at any other.

Stack L blocks and every token gets L rounds of "gather context, then think about it." That is the architecture. The rest of this document covers how sequences get in and out, and what happens when you train and run it.

## 4. Getting sequences in and out

### 4.1 Tokens, not words

Models don't see characters or words; they see **tokens** from a fixed vocabulary, usually built by byte-pair encoding (BPE). Common words are single tokens, rare words are split into pieces, and vocabularies range from ~32k (Llama 2) to 128k–200k (recent models). In English, a token is about three-quarters of a word on average.

**Practitioner note.** Tokenization explains a surprising share of model quirks. Arithmetic is hard partly because numbers tokenize inconsistently; spelling and letter-counting are hard because the model never sees letters; non-English text costs more tokens, and therefore more money and context, because vocabularies are English-skewed. When a model can't count the r's in "strawberry", that's tokenization, not stupidity.

### 4.2 Embeddings

A learned table of shape (vocab × d_model) maps each token id to its initial vector — the starting contents of that token's residual stream. At the output, a linear map of shape (d_model × vocab), the "unembedding" or LM head, turns the final stream into a score (logit) for every vocabulary entry; softmax turns those into next-token probabilities. Some models share the two matrices (GPT-2, Gemma); the Llama family keeps them separate.

### 4.3 Position: attention has no sense of order

Nothing in Sections 2 or 3 depends on token order. Attention is a weighted average over a *set*; shuffle the input tokens and the outputs shuffle identically. A Transformer without positional information is a bag-of-words model.

So position has to be injected. Three generations of solutions:

- **Sinusoidal encodings** (original paper): add a fixed vector of sines and cosines at different frequencies to each token's embedding. Unique per position, similar for nearby positions.
- **Learned absolute positions** (GPT-2, BERT): an embedding table indexed by position. Simple and effective, but undefined beyond the training length.
- **Rotary position embeddings, RoPE** (Su et al., 2021; used by Llama and most current open models): instead of adding anything to the embedding, rotate each query and key vector by an angle proportional to its position, arranged so that q·k depends only on the *relative* distance between the two tokens. Attention cares about "how far apart", not "where in absolute terms", and encoding that directly extrapolates better to longer contexts (with some frequency-scaling tricks).

You don't need RoPE's trigonometry. You need to know that position is a design choice bolted onto an order-blind core, and that the choice governs how well a model handles lengths it wasn't trained on.

## 5. Three flavors, and why one won

The original Transformer had an **encoder** (reads the source sentence; every token attends to every other) and a **decoder** (generates the target one token at a time, attending to the encoder's output and to its own earlier outputs). This encoder–decoder shape was built for translation.

Two simplifications followed:

- **Encoder-only** (BERT, 2018): just the encoder, trained by masking random tokens and predicting them. Every token sees every other, in both directions. Excellent for classification, embeddings, and extraction — anything where you have the whole input and want a representation of it.
- **Decoder-only** (GPT, 2018 onward): just the decoder's self-attention, trained to predict the next token. Each token may attend only to tokens *before* it, enforced by the **causal mask**: set the score for every future position to −∞ before the softmax, so its weight is exactly zero.

Decoder-only won the scaling race, and the reason matters. Next-token prediction gives a training signal at *every* position of *every* sequence, with no labels and no distinction between input and output. Any task — translation, summarization, classification, code — can be phrased as "here is some text; continue it." One objective, one architecture, effectively unlimited data. Encoder–decoder models (T5; the systems behind most machine translation and speech recognition) remain excellent when the task has a clear input→output structure. For general-purpose models, decoder-only is the default.

For the rest of this primer, "Transformer" means decoder-only unless stated otherwise.

## 6. Training

### 6.1 The objective

Take a long sequence of tokens. At each position t, the model outputs a distribution over the token at t+1; the loss is cross-entropy against the token that actually came next, averaged over all positions. Pretraining an LLM is running this over trillions of tokens.

The causal mask is what makes it efficient. Because position t sees only positions ≤ t, one forward pass over a sequence of length n produces n next-token predictions, each conditioned on exactly the right prefix. You get n training examples for the price of one pass, computed in parallel. (During training the model always sees the true prefix, never its own predictions — "teacher forcing.") An RNN yields the same n examples but computes them serially. This parallelism, more than any representational advantage, is why Transformers scaled and RNNs didn't.

### 6.2 What "scale" means

Three quantities determine what a pretrained model can do:

- **N**, parameters, set by d_model, L, and vocabulary size.
- **D**, training tokens.
- **C**, compute, well approximated by **C ≈ 6·N·D** FLOPs: 2 per parameter per token for the forward pass, 4 for the backward.

Scaling laws (Kaplan et al., 2020; Hoffmann et al., "Chinchilla", 2022) showed that loss falls as a smooth power law in each, and that for a fixed compute budget there's an optimal balance — roughly 20 tokens per parameter if training cost is all you care about. In practice models are now trained far past that (Llama-3 8B on ~15 trillion tokens, nearly 2,000 per parameter), because a smaller model trained longer is cheaper to *serve*, and serving cost dominates over a model's lifetime.

**Practitioner note.** This is why architecture debates feel low-stakes to people who train large models. The block has changed in details since 2017 (Section 9) but not in kind; nearly all the capability gain came from N, D, data quality, and training procedure. When someone proposes an architectural change, the first question is whether it moves the scaling curve or just the constant in front of it.

### 6.3 Post-training is not architecture

Instruction following, chat behavior, refusals, tool use — none of it is architectural. It comes from further training of the same network: supervised fine-tuning on curated examples, then preference optimization (RLHF, DPO, and successors), and increasingly reinforcement learning on tasks with checkable answers. The pretrained model is a document-continuation engine; post-training shapes what it continues into. Keep the two separate in your head; much of the confusion about what a model "knows" versus what it "does" dissolves when you do. How those post-training steps work — policy gradients, reward models and DPO, GRPO with verifiable rewards — is §1–4 of the [RL and thinking-models primer](../../rl-and-thinking-models/PRIMER.md#1-from-pretraining-to-post-training).

## 7. Inference: where practitioner intuition matters most

### 7.1 Autoregressive decoding

At inference the model generates one token at a time: run the forward pass on the prompt, sample a token from the distribution at the last position, append it, run again. There is no internal state between calls other than the tokens themselves. The model's memory is its context window and nothing else.

### 7.2 The KV cache, and why decoding is slow

Naively, generating token 1000 means re-running attention over all 1000 tokens, even though the keys and values for tokens 1–999 were already computed last step. So you cache them. Each new token computes only its own Q, K, V; its query attends over the cached K and V of everything before it; its own K and V join the cache.

This splits generation into two very different phases:

- **Prefill:** process the whole prompt in one parallel pass and populate the cache. Compute-bound; fast per token.
- **Decode:** one token per forward pass. Each step must read *every weight in the model* from GPU memory to produce a single token — a matmul with batch size 1, with terrible arithmetic intensity. Decode is **memory-bandwidth-bound**, not compute-bound.

Concretely: a 7B-parameter model in 16-bit precision is ~14 GB of weights. Generating one token means streaming those 14 GB through the memory bus. At 2–3 TB/s, that's 5–7 ms per token for a single sequence, with the compute units mostly idle. Serving systems recover efficiency by **batching**: one weight read serves dozens or hundreds of sequences at once.

This single fact explains most of the economics of LLM serving: why output tokens are priced well above input tokens; why throughput rises steeply with batch size; why quantizing weights to 8 or 4 bits speeds up decode even though it doesn't reduce FLOPs; why time-to-first-token and tokens-per-second are separate metrics with separate bottlenecks; and why speculative decoding works (a forward pass over five draft tokens costs about the same as over one, so a small model can propose and the large one can verify in a single pass).

### 7.3 The cost of context

The KV cache is not free. Per token it stores 2 × L × d_model numbers (one key and one value vector per layer). For Llama-2 7B in 16-bit that's about 0.5 MB per token: a 4k-token conversation costs ~2 GB of GPU memory, and 32k would cost ~16 GB, comparable to the weights themselves. That memory competes directly with batch size. Grouped-query attention (Section 9) exists largely to shrink this number.

Attention compute also grows with n² in sequence length. Whether the quadratic term dominates depends on n relative to d_model: for a 4096-wide model, the linear-in-n matmuls (projections and MLP) outweigh the n² score computation until sequences reach the tens of thousands of tokens. What bites earlier is memory — the n × n score matrix per head per layer. FlashAttention (2022) computes attention in tiles without ever materializing that matrix, which is what made long contexts practical. The math didn't change; the memory-access pattern did.

### 7.4 Decoding settings are not the model

Temperature, top-p, top-k, repetition penalties: these act on the output distribution *after* the architecture has finished. Temperature divides the logits before the softmax — lower sharpens the distribution, higher flattens it. None of it changes what the model computed. When output quality shifts with these settings, you've changed the sampling, not the model.

## 8. Where the numbers go

### 8.1 Parameter accounting

Per block, ignoring biases and norms (both negligible):

| Component | Parameters |
|---|---|
| Attention: W_Q, W_K, W_V, W_O | 4·d² |
| MLP with 4× expansion: W_up, W_down | 8·d² |
| **Per block** | **≈ 12·d²** |

Plus embeddings: vocab × d, doubled if input and output embeddings aren't tied.

Check against GPT-2 small (d = 768, L = 12, vocab = 50,257, tied embeddings, learned positions): 12 × 12 × 768² ≈ 85M, plus 50,257 × 768 ≈ 39M, plus 1,024 × 768 positions ≈ 0.8M. Total ≈ 124M. ✓

Llama-2 7B (d = 4096, L = 32, SwiGLU MLP with hidden size 11,008 and therefore three matrices, untied embeddings, vocab = 32,000): attention 4 × 4096² ≈ 67M, MLP 3 × 4096 × 11,008 ≈ 135M, so ≈ 202M per block, × 32 ≈ 6.5B, plus 2 × 32,000 × 4096 ≈ 0.26B. Total ≈ 6.7B. ✓

Doing this from a config file is a real skill. It gives you memory footprint (parameters × bytes per parameter), inference cost (≈ 2·N FLOPs per token, plus attention), and a sense of where a model's capacity sits.

### 8.2 Reading a model config

Hugging Face `config.json` fields map directly onto this primer:

| Field | Meaning |
|---|---|
| `hidden_size` | d_model, the residual stream width |
| `num_hidden_layers` | L, number of blocks |
| `num_attention_heads` | h; head dimension = hidden_size / h |
| `num_key_value_heads` | K/V heads for GQA; equals h for standard attention |
| `intermediate_size` | MLP hidden width |
| `vocab_size` | rows in the embedding table |
| `max_position_embeddings` | context length trained for |
| `rope_theta` | RoPE base frequency; larger values for long-context variants |
| `tie_word_embeddings` | whether input and output embeddings share weights |

### 8.3 Shapes through one forward pass

GPT-2 small, one sequence of n tokens:

    token ids                     (n,)
    embeddings + positions        (n, 768)
    per head: Q, K, V             (n, 64)        × 12 heads
    per head: attention scores    (n, n)         × 12 heads
    attention output (concat)     (n, 768)
    MLP hidden                    (n, 3072)
    block output                  (n, 768)       same shape in and out, × 12 blocks
    logits                        (n, 50257)

The residual stream shape never changes. Every block is a function from (n, d) to (n, d), which is why stacking any number of them is trivial.

## 9. Modern variants and why each exists

The 2017 design is still recognizable in every frontier model. What changed is a set of refinements, each fixing a specific problem:

| Change | Replaces | Why |
|---|---|---|
| Pre-norm | Post-norm | Stable training at depth without fragile warmup |
| RMSNorm | LayerNorm | Same effect, drops the mean-centering, cheaper |
| RoPE | Learned absolute positions | Relative position; better length extrapolation |
| SwiGLU / GeGLU | ReLU or GELU MLP | Gated activation reaches lower loss at equal compute; three matrices with ~2.7× hidden width keep the parameter count matched |
| No bias terms | Biases everywhere | Slightly more stable at scale, fewer parameters, no measurable loss |
| Grouped-query attention (GQA) | One K/V pair per head | Several query heads share one K/V head; cuts the KV cache 4–8× at small quality cost. Multi-query (MQA) is the extreme: one K/V for all heads |
| FlashAttention | Naive attention kernel | Identical math; tiled to avoid materializing the n × n matrix; large speed and memory wins |
| Mixture of Experts (MoE) | One MLP per block | E MLPs per block and a router that sends each token to the top-k; parameters grow ~E× while compute per token barely moves. More knowledge per FLOP, at the cost of memory and serving complexity; worked in depth in [mixture-of-experts](../../mixture-of-experts/PRIMER.md) |
| Sliding-window attention | Full attention in every layer | Some layers attend only to the last w tokens, often interleaved with full-attention layers; bounds cost on long inputs |
| Latent / compressed K/V (e.g. multi-head latent attention) | Standard K/V | Project K and V through a low-rank bottleneck; further KV cache reduction |

A "Llama-style" model — pre-norm RMSNorm, RoPE, SwiGLU, GQA, no biases, BPE tokenizer — is the recipe most open-weight models follow. If you can read one of those, you can read nearly all of them.

**Beyond text.** The same block handles images by cutting an image into patches, flattening each into a vector, and treating patches as tokens (Vision Transformer, 2020). Audio, video, and protein sequences work the same way. Most multimodal LLMs attach an encoder to a text model and feed its outputs in as extra tokens. The Transformer's defining property isn't that it understands language; it's that it's a general-purpose, parallelizable set-to-set function that scales. Anything you can tokenize, it can model.

## 10. Intuitions practitioners learn the hard way

**Attention weights are not explanations.** They're mixing coefficients in one head of one layer. A head attending strongly to a token doesn't mean the model is "reasoning about" that token, and heads with diffuse patterns can matter more than sharp ones. Treat attention maps as a diagnostic, not a rationale.

**The model is stateless.** Every call is a fresh forward pass over whatever's in the context. No memory across calls, no learning at inference time. The KV cache is a cache of computation, not a memory system. Anything you want the model to use at inference time has to be in the weights or in the context window.

**Depth is a fixed compute budget per token.** L blocks means each token gets exactly L sequential steps of processing, however hard the question. This is why chain-of-thought helps: writing intermediate tokens lets the model spend more forward passes, and more attention over its own scratch work, on one problem. "Thinking" models institutionalize the same move. When a model fails a task in one step that it can do in ten, the limit is often this fixed depth, not missing knowledge.

**In-context learning is a learned circuit, not a magic property.** The best-studied example is the *induction head*: two attention heads that together implement "find the last time this token appeared and copy what followed it." It emerges early in training and accounts for much of a model's ability to pick up patterns from a few examples in the prompt. The architecture doesn't contain this; training discovers it. Much of what looks like reasoning is a stack of learned circuits like this one.

**Long context is neither free nor fully used.** Doubling the window doubles KV memory and raises attention cost, and models attend to it unevenly, often favoring the beginning and end ("lost in the middle"). A 128k window is a capacity, not a promise that 128k tokens carry equal weight.

**Tokens are the unit of everything.** Cost, context, latency, and a whole class of failure modes are denominated in tokens. Learn to think in them.

**The architecture is rarely the interesting variable.** In most applied work, data, post-training, prompting, retrieval, and decoding strategy dominate outcomes. Understanding the architecture matters because it tells you what's *possible* and what things *cost*, not because you'll be changing it.

## 11. A working implementation

GPT-2's architecture in about fifty lines of PyTorch, minus dropout and initialization details. It runs. Reading it once, slowly, teaches more than reading this primer twice.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)   # W_q, W_k, W_v fused into one matmul
        self.out = nn.Linear(d, d, bias=False)        # W_o

    def forward(self, x):                             # x: (B, T, d)
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        # (B, T, d) -> (B, h, T, dh): each head gets a slice of the features
        q, k, v = (t.view(B, T, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
        scores = q @ k.transpose(-2, -1) / self.dh ** 0.5           # (B, h, T, T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))           # causal: no looking ahead
        w = F.softmax(scores, dim=-1)
        y = (w @ v).transpose(1, 2).reshape(B, T, d)                # concatenate heads
        return self.out(y)

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.up, self.down = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))

class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn, self.mlp = Attention(d, n_heads), MLP(d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))    # pre-norm, then residual add
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab, d=768, n_layers=12, n_heads=12, max_len=1024):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight    # weight tying

    def forward(self, idx):                   # idx: (B, T) token ids
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))        # logits: (B, T, vocab)
```

`GPT(vocab=50257)` has 124.4M parameters — GPT-2 small. Training is one more line. For a batch of token ids `idx` of shape (B, T):

```python
logits = model(idx)
loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                       idx[:, 1:].reshape(-1))
```

Every position's prediction is scored against the token that actually came next. One practical detail: with PyTorch's default initialization the starting loss is far above ln(vocab) ≈ 10.8; GPT-2 initializes weights with standard deviation 0.02, which brings the initial loss to roughly that value. A well-initialized language model starts out predicting a near-uniform distribution, and a starting loss far from ln(vocab) is a common early bug signal.

Swap `nn.LayerNorm` for RMSNorm, the position table for RoPE, the MLP for SwiGLU, and share K/V across groups of heads, and you have a modern open-weight model.

## 12. Glossary

- **Residual stream** — the (n × d_model) matrix flowing through the network; every block adds to it.
- **Head** — one independent attention pattern; a block runs h of them in parallel.
- **Causal mask** — position t attends only to positions ≤ t. Makes next-token training parallel and generation consistent.
- **Logits** — raw, pre-softmax scores over the vocabulary.
- **KV cache** — stored keys and values from earlier positions, so decoding doesn't recompute them.
- **Prefill / decode** — the parallel prompt-processing phase and the one-token-at-a-time generation phase.
- **Context window** — the maximum sequence length the model was trained to handle.
- **Pretraining / post-training** — next-token prediction on raw text; then fine-tuning to shape behavior.
- **GQA / MQA** — attention variants that share key/value heads to shrink the KV cache.
- **MoE** — mixture of experts; several MLPs per block, with a router choosing a few per token.
- **RoPE** — rotary position embedding; encodes relative position by rotating Q and K.

## 13. Further reading, in order

1. Jay Alammar, *The Illustrated Transformer* — the standard visual walkthrough.
2. Andrej Karpathy, *Let's build GPT: from scratch, in code, spelled out* (video) and the `nanoGPT` repository — Section 11 expanded into a full training run.
3. Vaswani et al., *Attention Is All You Need* (2017) — short and readable once you have the picture.
4. Harvard NLP, *The Annotated Transformer* — the paper as executable code.
5. Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla, 2022) — the scaling intuitions in Section 6.
6. Dao et al., *FlashAttention* (2022) — why memory access, not FLOPs, governs attention cost.
7. Olsson et al., *In-context Learning and Induction Heads* (2022) — what a learned circuit looks like up close.
8. Touvron et al., *Llama 2* (2023) — a complete, documented modern recipe.
