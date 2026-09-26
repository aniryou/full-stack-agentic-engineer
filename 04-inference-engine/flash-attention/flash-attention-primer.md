# FlashAttention: A Primer

**Assumed background:** you know what a matrix multiply is and have seen the word "transformer." Nothing else. Everything about attention, GPUs, and the algorithm itself is built up from scratch.

---

## The one-paragraph version

Attention is the core operation in a transformer. The textbook way to implement it is correct but wasteful: it builds a huge intermediate matrix, parks it in the GPU's slow main memory, and then reads it back twice. FlashAttention computes the *exact same result* without ever building that matrix, by processing the input in small tiles that fit in the GPU's fast on-chip memory. The hard part is that softmax needs to see a whole row before it can normalize — FlashAttention gets around this with a running-statistics trick called **online softmax**. The payoff is 2–4× wall-clock speedup and memory that grows linearly rather than quadratically with sequence length, which is a large part of why long-context models are practical at all.

The key thing to internalize: **this is not an approximation.** It is a different way of scheduling the same arithmetic.

---

## 1. What attention actually computes

You have a sequence of `N` tokens. Each token is projected into three vectors of dimension `d`:

- **Q** (query) — "what am I looking for?"
- **K** (key) — "what do I contain?"
- **V** (value) — "what do I pass along if selected?"

Stack them and you get three matrices of shape `N × d`. The computation is three steps:

```
S = Q @ Kᵀ           # N×N   raw similarity scores, every token against every token
P = softmax(S)       # N×N   normalize each row into a probability distribution
O = P @ V            # N×d   each output is a weighted average of value vectors
```

That's it. (Real implementations scale `S` by `1/√d` and often apply a causal mask so a token can't attend to the future, but the shape of the problem is unchanged.)

Note the shapes. `Q`, `K`, `V`, and `O` are all `N × d`, and `d` is small — typically 64 or 128 per attention head. But `S` and `P` are `N × N`. At `N = 8192`, `S` in fp16 is 128 MB **per head, per sequence in the batch**. With 32 heads and a batch of 8, that's 32 GB of intermediates for one layer. This is the whole problem in one sentence.

---

## 2. The hardware fact everything hinges on

A GPU has a memory hierarchy. Roughly, for an H100:

| Level | Capacity | Bandwidth | What it is |
|---|---|---|---|
| Registers | ~256 KB per SM | ~100+ TB/s | Per-thread scratch |
| SRAM (shared memory / L1) | ~256 KB per SM, ~33 MB total | ~30 TB/s | On-chip, software-managed |
| HBM (global memory) | 80 GB | ~3.3 TB/s | Off-chip stacked DRAM — "GPU memory" |

Bigger is slower, by roughly an order of magnitude per step. This is the same story as L1/L2/DRAM on a CPU, except the on-chip level is explicitly programmer-managed rather than a transparent cache.

Now the crucial ratio. An H100 does about 990 TFLOP/s of bf16 matrix math but moves only about 3.3 TB/s from HBM. Divide:

```
990e12 FLOP/s ÷ 3.3e12 byte/s ≈ 300 FLOPs per byte
```

**To keep the tensor cores busy, a kernel must do ~300 arithmetic operations for every byte it pulls from HBM.** Below that threshold the chip is starved: it sits idle waiting on memory, and adding more FLOPs is free. This ratio is called *arithmetic intensity*, and plotting achievable performance against it gives you the classic *roofline* — a diagonal memory-bound region rising to a flat compute-bound ceiling.

The ratio has gotten worse every generation, because compute has scaled faster than bandwidth. On an A100 it was ~200. On a B200, tensor core throughput roughly doubled again while HBM bandwidth did not keep pace. Increasingly, the bottleneck in ML kernels is not math — it's data movement.

---

## 3. Why the naive implementation is slow

The three-line version above, executed literally, does this:

1. Compute `S = Q @ Kᵀ`. Write `N×N` floats to HBM.
2. Read `S` back from HBM. Compute softmax. Write `P` (`N×N`) to HBM.
3. Read `P` back from HBM. Compute `P @ V`. Write `O`.

Let's price it out for one head at `N = 4096`, `d = 64`, fp16:

- **Useful work:** two matmuls, `2 × (2·N²·d)` ≈ **4.3 GFLOP**
- **Memory traffic:** `Q,K,V` are only 1.6 MB total, but `S` and `P` cost 33.5 MB each and get written and read — roughly **134 MB** of HBM traffic

Arithmetic intensity: `4.3e9 / 134e6` ≈ **32 FLOPs per byte**. Against a threshold of ~300, this kernel is running at roughly a tenth of the machine's capability. It is almost entirely memory-bound.

And it gets worse in the details. The two matmuls are efficient tensor-core work. But softmax, masking, dropout, and scaling are *elementwise* operations — one or two FLOPs per element — so they have terrible arithmetic intensity on their own. In a naive implementation each of them is a separate kernel launch that streams the entire `N×N` matrix out of HBM and back. The matmuls aren't the problem. The trips to memory between them are.

There's a second, harder failure: **you may simply run out of memory.** Peak memory is `O(N²)`. Double the context and you quadruple the footprint. This is the wall that long-context work runs into first.

---

## 4. The idea: tile it, and never write S

The fix is the standard one for memory-bound problems — *fusion* and *tiling*. Don't run softmax as a separate pass over a matrix in HBM. Chop the problem into blocks small enough to fit in SRAM, and do the whole score → softmax → weighted-sum chain on each block while it's still on-chip. `S` and `P` exist only as small tiles in fast memory and are discarded as soon as they're consumed.

Concretely: split `Q` into row blocks of size `Bᵣ × d` and `K`, `V` into blocks of size `Bᶜ × d`, sized so a few of them fit in a single SM's ~200 KB of shared memory. For each `Q` block, loop over all `K`/`V` blocks, accumulating into an output block.

This works fine for the matmuls, which decompose into blocks naturally. **The obstacle is softmax.**

Softmax normalizes across an entire row:

```
softmax(s)ᵢ = exp(sᵢ) / Σⱼ exp(sⱼ)
```

That denominator sums over the whole row — all `N` entries. If you're holding a block of 128 columns, you can't compute it. You'd seem to need a full pass over the row before you can emit any output for it, which is exactly the thing you were trying to avoid.

(There's also a numerical wrinkle: `exp` of a large score overflows fp16 immediately, so every real implementation subtracts the row max first — `exp(sᵢ - max)`. Mathematically identical, numerically essential. And now you need the row *max* up front too, which is a second global reduction.)

---

## 5. Online softmax: the actual trick

The resolution is to compute softmax **incrementally**, maintaining running statistics and retroactively correcting earlier work as new blocks arrive. (The technique predates FlashAttention — Milakov & Gimelshein, 2018 — but FlashAttention is what made it matter.)

For each row of the output, carry three running values:

- `m` — the largest score seen so far
- `ℓ` — the running sum of `exp(score − m)`
- `O` — the running *unnormalized* output accumulator

When a new block of scores `S_j` arrives:

```
m_new = max(m_old, rowmax(S_j))
α     = exp(m_old − m_new)                          # correction factor
ℓ_new = α · ℓ_old + rowsum(exp(S_j − m_new))
O_new = α · O_old + exp(S_j − m_new) @ V_j
```

After the last block, divide once: `O = O_new / ℓ_new`.

The correction factor `α` is the entire idea. If a later block contains a bigger score, everything accumulated so far was exponentiated against a stale, too-small max — so it's off by exactly a factor of `exp(m_old − m_new)`. Multiply through and the books balance.

### A worked example

Take one row with four scores, arriving in two blocks of two: `[1, 3]` then `[5, 2]`.

**The answer we're aiming for.** Global max is 5. `exp([1,3,5,2] − 5) = [0.0183, 0.1353, 1.0, 0.0498]`, summing to `1.2034`. So the true weights are `[0.0152, 0.1125, 0.8310, 0.0414]`.

**Block 1.** `m = 3`, `ℓ = exp(1−3) + exp(3−3) = 0.1353 + 1 = 1.1353`, and `O = 0.1353·v₁ + 1.0·v₂`. If we stopped here we'd get the softmax of just the first two elements — correct for what we've seen, wrong overall.

**Block 2.** New max is 5, so `m_new = 5` and `α = exp(3−5) = 0.1353`.

```
ℓ_new = 0.1353 × 1.1353 + (exp(5−5) + exp(2−5))
      = 0.1536 + 1.0498
      = 1.2034                                    ✓ matches the global denominator

O_new = 0.1353 × (0.1353·v₁ + 1.0·v₂) + 1.0·v₃ + 0.0498·v₄
      = 0.0183·v₁ + 0.1353·v₂ + 1.0·v₃ + 0.0498·v₄  ✓ matches the global numerators
```

Divide by `ℓ_new = 1.2034` and you recover `[0.0152, 0.1125, 0.8310, 0.0414]` exactly.

No approximation anywhere — just deferred normalization plus bookkeeping. The `N×N` score matrix never existed in full.

---

## 6. The backward pass: recompute instead of store

Training needs gradients, and the textbook backward pass for attention needs `P` — the very `N×N` matrix we just refused to build. Storing it would put the quadratic memory cost straight back.

FlashAttention instead **recomputes** it. During the forward pass it saves only `O` (`N×d`) and the final softmax statistics per row, packed as the log-sum-exp `L = m + log(ℓ)` — that's `N` numbers, not `N²`. In the backward pass it reloads `Q`, `K`, `V` tiles into SRAM and reconstructs each `S` and `P` tile on the fly, using the saved `L` to normalize correctly without a second reduction pass.

This costs *more* FLOPs than storing `P` would. It is still faster, because the kernel was never compute-bound to begin with — you're spending surplus arithmetic to buy back scarce bandwidth. It's the same logic as gradient checkpointing, applied surgically at the level of a single fused kernel rather than whole layers.

---

## 7. What you get

- **Memory:** `O(N)` extra storage per head instead of `O(N²)`. This is the change that unlocks long context.
- **HBM traffic:** roughly `Θ(N²d²/M)` accesses (`M` = SRAM size) versus `Θ(N² + Nd)` — the original paper measured up to 9× fewer accesses on GPT-2 shapes.
- **Wall clock:** typically 2–4× on the attention layer, more at long sequence lengths.
- **Exactness:** bit-for-bit-equivalent mathematics. Floating-point non-associativity means results won't be *bitwise* identical to a naive implementation (or between different tile configurations), but there's no approximation error in the algorithmic sense.

That last point deserves emphasis. Around 2020–2021 there was a wave of "efficient attention" work — Linformer, Performer, sparse and low-rank schemes — that reduced the `O(N²)` cost by *changing the computation*, accepting some quality loss, and often failing to deliver real wall-clock wins because they traded matmuls for memory-bound gather operations. FlashAttention won by leaving the math alone and fixing the memory schedule instead. The lesson generalized well beyond attention.

---

## 8. The lineage

**FlashAttention-1** (Dao et al., 2022) — tiling, online softmax, backward recomputation. Established the approach.

**FlashAttention-2** (2023) — mostly a work-partitioning rewrite. Three changes mattered: swapping the loop order so the outer loop runs over `Q` blocks (each output block is then owned by one thread block, no cross-block accumulation); parallelizing over the sequence dimension, not just batch × heads, which matters when batch is small and `N` is large; and minimizing non-matmul FLOPs. That last one is subtler than it sounds — on an A100, tensor-core matmul runs at 312 TFLOP/s while general FP32 arithmetic runs at 19.5, so a non-matmul FLOP costs ~16× a matmul FLOP. Deferring the rescaling division to the end of the loop instead of doing it per block is worth real percentage points. Result: ~2× over FA1, ~70% of A100 peak (paper figures, verify).

**FlashAttention-3** (2024) — Hopper-specific. Exploits asynchrony: TMA for background global↔shared copies, warp specialization into producer (loading) and consumer (computing) roles, and a ping-pong schedule that overlaps the softmax of one block with the matmul of another so the slow exponential units hide behind the tensor cores. Adds FP8 with incoherent processing (a Hadamard rotation to spread outliers before quantization). Reached roughly 740 TFLOP/s at 75% utilization on H100 (paper figure, verify).

**FlashAttention-4** (paper published March 2026) — Blackwell. The framing is what the authors call **asymmetric hardware scaling**: tensor core throughput doubles while other functional units — shared memory bandwidth, exponential units — scale more slowly or not at all. Dense BF16 tensor core throughput went from roughly 1 PFLOPS to 2.25 PFLOPS (datasheet, verify), and the hardware added new tensor core instructions (TCGEN05), a new memory space called Tensor Memory for tensor core intermediates, and a fully asynchronous MMA execution model. So the bottleneck moved, and the kernel had to move with it. The responses are worth knowing because they're so specific: a software emulation of the exponential via polynomial approximation on the FMA units, to relieve pressure on the dedicated exponential hardware, plus conditional online softmax rescaling; and on the backward pass, storing intermediates in tensor memory to relieve shared-memory traffic combined with Blackwell's 2-CTA MMA mode. Two query tiles of 128 tokens each are computed per CTA and alternated in a ping-pong schedule. It's also written entirely in CuTe-DSL embedded in Python, with 20–30× faster compile times than C++ template-based approaches (paper figure, verify) — anyone who has waited on a `flash-attn` build will appreciate that. Performance, as the paper reports it (verify): up to 1605 TFLOP/s on B200 with BF16, 71% utilization, 1.3× faster than cuDNN 9.13 and 2.7× faster than Triton.

One instructive footnote from the FA4 rollout (verify): for *inference decode*, FlashAttention-4 was initially slower than FlashAttention-2 on B200s until split-KV was ported over. Generating one token at a time is a different regime — a single query row against a long KV cache leaves most SMs idle unless you split along the key dimension and reduce afterwards. Worth remembering that "the fastest attention kernel" is always shape-dependent.

---

## 9. The descendants

The online-softmax accumulator turned out to be a reusable primitive. Once you know that partial attention results can be merged with a rescaling factor, several other things fall out:

- **Flash-Decoding / split-KV** — split the KV cache across SMs during autoregressive decode, merge with the same correction. Essential for long-context inference.
- **Ring Attention** — run the same accumulation across *devices*, passing KV blocks around a ring while overlapping communication with compute. Extends context length beyond a single GPU's memory.
- **PagedAttention** (vLLM) — orthogonal, but complementary: virtual-memory-style paging of the KV cache to eliminate fragmentation.
- **FlexAttention** (PyTorch) — write a custom `score_mod` or `mask_mod` in a few lines of Python and get a compiled flash-style kernel, so you don't need bespoke CUDA for every masking variant. Now backed by FA4 on Blackwell (verify).

---

## 10. Practical notes

**How you actually invoke it.** Most of the time you don't write it yourself — `torch.nn.functional.scaled_dot_product_attention` dispatches to a flash kernel when the shapes and dtypes qualify. For direct control, `flash_attn_func` and the `varlen` variants from the `flash-attn` package. Serving stacks (vLLM, SGLang, TensorRT-LLM) have it wired in already.

**Things that trip people up:**

- It reduces *memory traffic*, not FLOPs. The arithmetic is still `O(N²d)`. If someone tells you FlashAttention made attention linear, they're wrong.
- Causal masking is worth roughly 2×, because fully-masked blocks above the diagonal can be skipped entirely. Make sure the flag is actually set.
- The fast paths want fp16/bf16 and constrained head dimensions. Silent fallback to a slower kernel is a common cause of "why didn't this help?"
- You cannot inspect the attention matrix afterward — it was never materialized. If you need attention maps for interpretability or visualization, you'll have to recompute them separately, at full quadratic cost.
- At short sequence lengths the win is small, because attention isn't your bottleneck there anyway.
- Prefill and decode are different problems. A kernel tuned for one can be worse than useless for the other.

---

## The mental model to keep

Attention is not expensive because of the math. It's expensive because the naive schedule shuttles a quadratic intermediate to and from slow memory three times. Tiling plus online softmax removes those trips without changing a single result. Every subsequent version is the same idea re-tuned as the hardware's bottleneck migrates — first bandwidth, then work partitioning, then asynchrony, and now the exponential units and shared memory falling behind the tensor cores.

If you take one transferable thing from this: **on modern accelerators, look at the memory schedule before you look at the FLOP count.**

---

## Going deeper

- Dao et al., *FlashAttention* (2022), *FlashAttention-2* (2023), *FlashAttention-3* (2024), *FlashAttention-4* (2026) — the four papers, readable in order.
- Milakov & Gimelshein, *Online normalizer calculation for softmax* (2018) — the trick, in isolation.
- Williams et al., *Roofline* (2009) — the performance model underneath all of this.
- The `Dao-AILab/flash-attention` repo — the CuTeDSL rewrite is far more approachable than the old C++ templates.

Worth doing on any machine, no GPU needed: implement the online-softmax accumulator yourself in ~50 lines of numpy (or PyTorch) and check it against a reference softmax. It takes an afternoon and the idea stops being abstract; [`flash_attention_minimal.py`](flash_attention_minimal.py) is one answer to compare with.

**Code and tests.** [`kernel-core`](../kernel-core/README.md)'s `kerncore.flash` is the same tiled forward with counters: it checks itself against `flash_attention_minimal.py`, counts tiles and bytes against [`fa_calculators.py`](fa_calculators.py) (which it imports), and shows why a forward key loop needs no `-inf` guard (deep dive §11.2); the tests are `kernel-core/tests/test_flash.py`, and `kernel-core/tests/test_primer_numbers.py` recomputes this page's worked numbers.

---

## Verify list (dated 2026-09-26)

Product and paper facts this primer states; every other number on the page is derived from them. The
[deep dive](flash-attention-deep-dive.md) re-derives most of them and keeps its own verify list.

| Item | Value used | Why it needs checking |
|---|---|---|
| H100 memory hierarchy (§2) | ~256 KB registers and ~256 KB shared memory/L1 per SM, ~33 MB SRAM total, 80 GB HBM at ~3.3 TB/s | rounded datasheet figures; the SXM part is 3.35 TB/s and 228 KB of it is usable shared memory |
| Peak bf16 matmul (§2) | H100 ~990 TFLOP/s; ridge ~300 FLOP/B; A100 ~200 FLOP/B (the 40 GB part) | dense datasheet peaks; the A100 80 GB (2.0 TB/s) gives ~153 |
| A100 matmul vs FP32 (§8) | 312 vs 19.5 TFLOP/s | datasheet |
| FA1 traffic saving (§7) | up to 9× fewer HBM accesses on GPT-2 shapes | FA1 paper |
| FA2 (§8) | ~2× over FA1, ~70% of A100 peak | FA2 paper; the deep dive gives the 50–73% range |
| FA3 (§8) | ~740 TFLOP/s bf16, 75% of H100 | FA3 paper |
| B200 tensor throughput (§8) | dense bf16 ~2.25 PFLOP/s, up from ~1 | datasheet |
| FA4 (§8) | paper March 2026; up to 1605 TFLOP/s bf16 on B200 (71%), 1.3× cuDNN 9.13, 2.7× Triton; 20–30× faster compiles with CuTe-DSL | FA4 paper; not fetched on this date |
| FA4 decode (§8) | initially slower than FA2 on B200 until split-KV was ported | project history; check the current release notes |
| FlexAttention (§9) | backed by FA4 on Blackwell | PyTorch release in use |
