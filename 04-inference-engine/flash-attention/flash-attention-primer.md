# FlashAttention: A Primer

**Assumed background:** You know what a matrix multiply is, and you saw the word "transformer" before. You do not need more. This primer builds all of the information about attention, GPUs and the algorithm from the start.

---

## The one-paragraph version

Attention is the core operation in a transformer. The textbook implementation is correct, but it wastes resources. It makes a large intermediate matrix, puts it in the slow main memory of the GPU, and then reads it back two times. FlashAttention calculates the *exact same result*, but it never makes that matrix. It does the work on small tiles of the input that fit in the fast on-chip memory of the GPU. The difficult part is that softmax must see a full row before it can normalize the row.

FlashAttention solves this problem with a trick that keeps statistics and updates them block by block. The name of this trick is **online softmax**. The result is a wall-clock speedup of 2–4×. Also, the memory increases linearly with sequence length, not quadratically. This is a large part of the reason that long-context models are practical at all.

The key point to remember: **this is not an approximation.** It is a different schedule for the same arithmetic.

---

## 1. What attention actually computes

You have a sequence of $N$ tokens. The model projects each token into three vectors of dimension $d$:

- **Q** (query): "what am I looking for?"
- **K** (key): "what do I contain?"
- **V** (value): "what do I pass along if selected?"

When you put these vectors for all tokens in rows, you get three matrices of shape $N \times d$. The calculation has three steps:

$$
\begin{aligned}
S &= QK^{\top} && N \times N && \text{raw similarity scores, every token against every token} \\
P &= \operatorname{softmax}(S) && N \times N && \text{normalize each row into a probability distribution} \\
O &= PV && N \times d && \text{each output is a weighted average of value vectors}
\end{aligned}
$$

That is all. (Real implementations scale $S$ by $1/\sqrt{d}$. They often also apply a causal mask, so that a token cannot attend to the future. But the shape of the problem does not change.)

Look at the shapes: $Q$, $K$, $V$, and $O$ are all $N \times d$. The value of $d$ is small, usually 64 or 128 per attention head. But $S$ and $P$ are $N \times N$. At $N$ = 8192, `S` in fp16 is 128 MB **per head, per sequence in the batch**. With 32 heads and a batch of 8, that's 32 GB of intermediates for one layer. This is the full problem in one sentence.

---

## 2. The hardware fact everything hinges on

A GPU has a memory hierarchy. These are the approximate values for an H100:

| Level | Capacity | Bandwidth | What it is |
|---|---|---|---|
| Registers | ~256 KB per SM | ~100+ TB/s | Scratch memory for each thread |
| SRAM (shared memory / L1) | ~256 KB per SM, ~33 MB total | ~30 TB/s | On-chip, software-managed |
| HBM (global memory) | 80 GB | ~3.3 TB/s | Off-chip stacked DRAM ("GPU memory") |

A larger level is slower, by approximately one order of magnitude per step. A CPU has the same structure with L1/L2/DRAM. But on a GPU, the programmer must control the on-chip level in the code. It is not a transparent cache.

Now look at the most important ratio. An H100 does approximately 990 TFLOP/s of bf16 matrix math. But it moves only approximately 3.3 TB/s from HBM. Divide the two numbers:

```
990e12 FLOP/s ÷ 3.3e12 byte/s ≈ 300 FLOPs per byte
```

**To keep the tensor cores busy, a kernel must do ~300 arithmetic operations for every byte that it reads from HBM.** Below that threshold, the chip does not get sufficient data. It stays idle while it waits for memory, and more FLOPs cost nothing. The name of this ratio is *arithmetic intensity*. Make a plot of the possible performance against this ratio, and you get the classic *roofline*. The roofline has a diagonal memory-bound region that goes up to a flat compute-bound ceiling.

The ratio became worse in each generation, because compute increased faster than bandwidth. On an A100, it was ~200. On a B200, the tensor core throughput approximately doubled again, but the HBM bandwidth did not increase at the same rate. More and more, the bottleneck in ML kernels is not the math. It is the movement of data.

---

## 3. Why the naive implementation is slow

When a program runs the three-line version of §1 literally, it does these steps:

1. Calculate $S = QK^{\top}$. Write $N \times N$ floats to HBM.
2. Read $S$ back from HBM. Calculate the softmax. Write $P$ ($N \times N$) to HBM.
3. Read $P$ back from HBM. Calculate ${PV}$. Write $O$.

Calculate the cost for one head at $N$ = 4096, $d$ = 64, fp16:

- **Useful work:** Two matmuls, $2 \times (2 \cdot N^2 \cdot d)$ ≈ **4.3 GFLOP**.
- **Memory traffic:** `Q,K,V` are only 1.6 MB total. But $S$ and $P$ cost 33.5 MB each, and the kernel writes and reads each of them. The total is roughly **134 MB** of HBM traffic.

Arithmetic intensity: `4.3e9 / 134e6` ≈ **32 FLOPs per byte**. Compared with the threshold of ~300, this kernel runs at approximately one tenth of the capability of the machine. It is almost fully memory-bound.

The details make it worse. The two matmuls are efficient work for the tensor cores. But softmax, the mask, dropout and the scale are *elementwise* operations, with one or two FLOPs per element. Thus, each of them alone has a low arithmetic intensity. In a simple implementation, each of them is a separate kernel launch that moves the full $N \times N$ matrix out of HBM and back. The problem is not the matmuls, but the trips to memory between them.

There is a second failure, and it is more difficult: **it is possible that the memory is not sufficient.** Peak memory is $O(N^2)$. If you double the context, the memory footprint becomes four times larger. Long-context work meets this limit first.

---

## 4. The idea: tile it, and never write S

The solution is the standard solution for memory-bound problems: *fusion* and *tiling*. Do not run softmax as a separate pass over a matrix in HBM. Divide the problem into blocks that are sufficiently small to fit in SRAM. Then do the full chain of score, softmax and weighted sum on each block while the block is still on-chip. $S$ and $P$ exist only as small tiles in fast memory. The kernel removes each tile immediately after it uses the tile.

In practice: divide $Q$ into row blocks of size $B_r \times d$, and divide $K$, $V$ into blocks of size $B_c \times d$. Select the sizes so that some of these blocks fit in the ~200 KB of shared memory of one SM. For each $Q$ block, loop over all $K$/$V$ blocks, and accumulate the results into an output block.

This is easy for the matmuls, because a matmul divides into blocks naturally. **The obstacle is softmax.**

Softmax normalizes across a full row:

$$
\operatorname{softmax}(s)_i = \frac{\exp(s_i)}{\sum_j \exp(s_j)}
$$

The denominator is a sum over the full row, that is, all $N$ entries. If you hold only a block of 128 columns, you cannot calculate it. It seems that you must do a full pass over the row before you can give any output for that row. That pass is exactly the thing that you tried to prevent.

(There is also a numerical problem. The $\exp$ of a large score overflows fp16 immediately. Thus, every real implementation subtracts the row max first: $\exp(s_i - \max)$. The result is mathematically identical, and the step is numerically necessary. Now you also need the row *max* at the start, and that is a second global reduction.)

---

## 5. Online softmax: the actual trick

The solution is to calculate softmax **incrementally**. The kernel keeps statistics and updates them as new blocks arrive, and it corrects the earlier work retroactively. (The technique is older than FlashAttention: Milakov and Gimelshein, 2018. But FlashAttention made it important.)

For each row of the output, keep three values that change as the blocks arrive:

- $m$: the largest score so far.
- $\ell$: the sum so far of $\exp(\text{score} - m)$.
- $O$: the *unnormalized* output accumulator so far.

When a new block of scores $S_j$ arrives:

$$
\begin{aligned}
m_{\text{new}} &= \max\big(m_{\text{old}},\ \operatorname{rowmax}(S_j)\big) \\
\alpha &= \exp(m_{\text{old}} - m_{\text{new}}) && \text{correction factor} \\
\ell_{\text{new}} &= \alpha \cdot \ell_{\text{old}} + \operatorname{rowsum}\big(\exp(S_j - m_{\text{new}})\big) \\
O_{\text{new}} &= \alpha \cdot O_{\text{old}} + \exp(S_j - m_{\text{new}}) \, V_j
\end{aligned}
$$

After the last block, divide one time: $O = O_{\text{new}} / \ell_{\text{new}}$.

The correction factor $\alpha$ is the full idea. If a later block contains a larger score, the kernel used an old max that was too small for all of the earlier exponentials. Thus, the accumulated values are incorrect by exactly a factor of $\exp(m_{\text{old}} - m_{\text{new}})$. Multiply all of them by this factor, and the totals are correct again.

### A worked example

Use one row with four scores. The scores arrive in two blocks of two: `[1, 3]` then `[5, 2]`.

**The answer we're aiming for.** The global max is 5. The exponentials are `exp([1,3,5,2] − 5) = [0.0183, 0.1353, 1.0, 0.0498]`, summing to `1.2034`. So the true weights are `[0.0152, 0.1125, 0.8310, 0.0414]`.

**Block 1.** `m = 3`, `ℓ = exp(1−3) + exp(3−3) = 0.1353 + 1 = 1.1353`, and `O = 0.1353·v₁ + 1.0·v₂`. If the kernel stops here, the result is the softmax of only the first two elements. That result is correct for the elements so far, but it is incorrect for the full row.

**Block 2.** The new max is 5. Thus `m_new = 5` and `α = exp(3−5) = 0.1353`.

```
ℓ_new = 0.1353 × 1.1353 + (exp(5−5) + exp(2−5))
      = 0.1536 + 1.0498
      = 1.2034                                    ✓ matches the global denominator

O_new = 0.1353 × (0.1353·v₁ + 1.0·v₂) + 1.0·v₃ + 0.0498·v₄
      = 0.0183·v₁ + 0.1353·v₂ + 1.0·v₃ + 0.0498·v₄  ✓ matches the global numerators
```

Divide by `ℓ_new = 1.2034`, and you recover `[0.0152, 0.1125, 0.8310, 0.0414]` exactly.

There is no approximation at any step. There is only a deferred normalization and a record of the statistics. The full $N \times N$ score matrix never existed.

---

## 6. The backward pass: recompute instead of store

Training needs gradients. The textbook backward pass for attention needs $P$, and $P$ is exactly the $N \times N$ matrix that the forward pass did not make. If the kernel stores $P$, the quadratic memory cost comes back immediately.

FlashAttention **recomputes** $P$ instead. During the forward pass, it keeps only $O$ ($N \times d$) and the final softmax statistics for each row. It packs the statistics as the log-sum-exp $L = m + \log(\ell)$, that is, $N$ numbers, not $N^2$. In the backward pass, it loads the $Q$, $K$, $V$ tiles into SRAM again. Then it calculates each $S$ and $P$ tile again when it needs the tile. It uses the saved $L$ to normalize correctly without a second reduction pass.

This method costs *more* FLOPs than a stored $P$. But it is still faster, because the kernel was never compute-bound. The kernel has extra arithmetic, but bandwidth is scarce. Thus, you use the extra arithmetic to get back bandwidth. It is the same logic as gradient checkpointing. But here, the logic applies precisely at the level of one fused kernel, not at the level of full layers.

---

## 7. What you get

- **Memory:** ${O(N)}$ extra storage per head, not $O(N^2)$. This is the change that makes long context possible.
- **HBM traffic:** approximately $\Theta(N^2 d^2/M)$ accesses ($M$ = SRAM size), against $\Theta(N^2 + Nd)$. The original paper measured up to 9× fewer accesses on GPT-2 shapes.
- **Wall clock:** usually 2–4× on the attention layer, and more at long sequence lengths.
- **Exactness:** The mathematics is bit-for-bit-equivalent. Floating-point arithmetic is not associative. Thus, the results will not be *bitwise* identical to a simple implementation (or between different tile configurations). But there is no approximation error in the algorithmic sense.

The last point is important. Around 2020–2021, there was a wave of "efficient attention" work: Linformer, Performer, and sparse and low-rank schemes. These methods decreased the $O(N^2)$ cost because they *changed the computation*, and they accepted some loss of quality. Often they did not give real wall-clock gains, because they replaced matmuls with memory-bound gather operations. FlashAttention won because it kept the math and repaired the memory schedule instead. The lesson was general, and it applied far beyond attention.

---

## 8. The lineage

**FlashAttention-1** (Dao et al., 2022): tiling, online softmax and backward recomputation. This version set up the approach.

**FlashAttention-2** (2023): mostly a rewrite that divides the work differently. Three changes were important:

- A swap of the loop order, so that the outer loop runs over $Q$ blocks. Then one thread block owns each output block, and there is no accumulation across blocks.
- Parallelism over the sequence dimension, not only over batch × heads. This is important when the batch is small and $N$ is large.
- As few non-matmul FLOPs as possible.

The last change is less obvious than it seems. On an A100, tensor-core matmul runs at 312 TFLOP/s, but general FP32 arithmetic runs at 19.5. Thus, a non-matmul FLOP costs ~16× a matmul FLOP. When the kernel does the division that rescales the output at the end of the loop, not in each block, it gains real percentage points. Result: ~2× over FA1, ~70% of A100 peak (paper figures, verify).

**FlashAttention-3** (2024): specific to Hopper. It uses asynchrony in three ways:

- TMA, for background copies between global and shared memory.
- Warp specialization into producer roles (they load the data) and consumer roles (they calculate).
- A ping-pong schedule that overlaps the softmax of one block with the matmul of another block. Thus, the slow exponential units hide behind the tensor cores.

It also adds FP8 with incoherent processing (a Hadamard rotation that spreads outliers before quantization). It reached approximately 740 TFLOP/s at 75% utilization on H100 (paper figure, verify).

**FlashAttention-4** (paper published March 2026): Blackwell. The authors call the main idea **asymmetric hardware scaling**. The tensor core throughput doubles, but other functional units scale more slowly or not at all. Examples are the shared memory bandwidth and the exponential units. Dense BF16 tensor core throughput went from approximately 1 PFLOPS to 2.25 PFLOPS (datasheet, verify). The hardware also added these items:

- New tensor core instructions (TCGEN05).
- A new memory space with the name Tensor Memory, for tensor core intermediates.
- A fully asynchronous MMA execution model.

Thus, the bottleneck moved, and the kernel had to move with it.

The responses are important to know, because they are specific:

- A software emulation of the exponential, with polynomial approximation on the FMA units. This decreases the load on the dedicated exponential hardware.
- A conditional rescale in the online softmax.
- On the backward pass, the kernel stores intermediates in tensor memory to decrease shared-memory traffic. It uses this together with the 2-CTA MMA mode of Blackwell.

Each CTA calculates two query tiles of 128 tokens each, and it alternates them in a ping-pong schedule.

The authors also wrote it fully in CuTe-DSL, embedded in Python. Its compile times are 20–30× faster than the compile times of approaches based on C++ templates (paper figure, verify). If you waited for a `flash-attn` build before, you will like this change. Performance, as the paper reports it (verify): up to 1605 TFLOP/s on B200 with BF16, at 71% utilization. That is 1.3× faster than cuDNN 9.13 and 2.7× faster than Triton.

One useful note from the FA4 rollout (verify): for *inference decode*, FlashAttention-4 was at first slower than FlashAttention-2 on B200s, until split-KV came to FA4. The generation of one token at a time is a different case. A single query row against a long KV cache leaves most SMs idle, unless you split along the key dimension and reduce after that. Remember that "the fastest attention kernel" always depends on the shape.

---

## 9. The descendants

It became clear that the online-softmax accumulator is a reusable primitive. When you know that a rescale factor can merge partial attention results, several other methods follow:

- **Flash-Decoding / split-KV**: split the KV cache across SMs during autoregressive decode, and merge the results with the same correction. This is necessary for long-context inference.
- **Ring Attention**: run the same accumulation across *devices*. The devices pass KV blocks around a ring, and they overlap communication with compute. This extends the context length beyond the memory of a single GPU.
- **PagedAttention** (vLLM): orthogonal, but complementary. It puts the KV cache in pages, in the style of virtual memory, to remove fragmentation.
- **FlexAttention** (PyTorch): write a custom `score_mod` or `mask_mod` in a few lines of Python, and you get a compiled flash-style kernel. Thus, you do not need custom CUDA for each mask variant. On Blackwell, it now uses FA4 as its backend (verify).

---

## 10. Practical notes

**How you actually invoke it.** Usually, you do not write it yourself. `torch.nn.functional.scaled_dot_product_attention` dispatches to a flash kernel when the shapes and dtypes are applicable. For direct control, use `flash_attn_func` and the `varlen` variants from the `flash-attn` package. Serving stacks (vLLM, SGLang, TensorRT-LLM) already contain it.

**Things that trip people up:**

- It decreases *memory traffic*, not FLOPs. The arithmetic is still $O(N^2 d)$. If a person tells you that FlashAttention made attention linear, that person is incorrect.
- Causal masking is worth roughly 2×, because the kernel can skip the fully-masked blocks above the diagonal. Make sure that the flag is set.
- The fast paths need fp16/bf16 and a limited set of head dimensions. A silent fallback to a slower kernel is a frequent cause of "why didn't this help?"
- You cannot examine the attention matrix after the run, because the full matrix never existed in memory. If you need attention maps for interpretability or visualization, you must recompute them separately, at the full quadratic cost.
- At short sequence lengths, the gain is small, because attention is not your bottleneck there in any case.
- Prefill and decode are different problems. A kernel adjusted for one of them can be worse than useless for the other.

---

## The mental model to keep

Attention is not high-cost because of the math. It is high-cost because the simple schedule moves a quadratic intermediate to and from slow memory three times. Tiling and online softmax remove those trips, and they do not change one result. Each later version is the same idea, adjusted again when the bottleneck of the hardware moves. The bottleneck was first bandwidth, then the division of work, then asynchrony. Now it is the exponential units and shared memory, which do not keep pace with the tensor cores.

The one general lesson to keep from this primer: **on modern accelerators, look at the memory schedule before you look at the FLOP count.**

---

## Going deeper

- Dao et al., *FlashAttention* (2022), *FlashAttention-2* (2023), *FlashAttention-3* (2024), *FlashAttention-4* (2026): the four papers. You can read them in order.
- Milakov and Gimelshein, *Online normalizer calculation for softmax* (2018): the trick, alone.
- Williams et al., *Roofline* (2009): the performance model under all of this.
- The `Dao-AILab/flash-attention` repo: the CuTeDSL rewrite is much easier to read than the old C++ templates.

Do this exercise on any machine. It needs no GPU. Write the online-softmax accumulator yourself in ~50 lines of numpy (or PyTorch). Compare it with a reference softmax. The work takes an afternoon, and after it the idea is no longer abstract. [`flash_attention_minimal.py`](flash_attention_minimal.py) is one answer to compare with yours.

**Code and tests.** The module `kerncore.flash` of [`kernel-core`](../kernel-core/README.md) is the same tiled forward, with counters. It compares itself with `flash_attention_minimal.py`. It counts tiles and bytes and compares them with [`fa_calculators.py`](fa_calculators.py) (which it imports). It also shows why a forward key loop needs no `-inf` guard (deep dive §11.2). The tests are `kernel-core/tests/test_flash.py`, and `kernel-core/tests/test_primer_numbers.py` recomputes the worked numbers of this page.

---

## Verify list (dated 2026-09-26)

This table lists the product and paper facts that this primer states. Every other number on the page comes from them. The
[deep dive](flash-attention-deep-dive.md) derives most of them again and keeps its own verify list.

| Item | Value used | Why it needs checking |
|---|---|---|
| H100 memory hierarchy (§2) | ~256 KB registers and ~256 KB shared memory/L1 per SM, ~33 MB SRAM total, 80 GB HBM at ~3.3 TB/s | Rounded datasheet figures. The SXM part is 3.35 TB/s, and 228 KB of it is usable shared memory. |
| Peak bf16 matmul (§2) | H100 ~990 TFLOP/s, ridge ~300 FLOP/B, A100 ~200 FLOP/B (the 40 GB part) | Dense datasheet peaks. The A100 80 GB (2.0 TB/s) gives ~153. |
| A100 matmul against FP32 (§8) | 312 against 19.5 TFLOP/s | datasheet |
| FA1 traffic reduction (§7) | up to 9× fewer HBM accesses on GPT-2 shapes | FA1 paper |
| FA2 (§8) | ~2× over FA1, ~70% of A100 peak | FA2 paper. The deep dive gives the 50–73% range. |
| FA3 (§8) | ~740 TFLOP/s bf16, 75% of H100 | FA3 paper |
| B200 tensor throughput (§8) | dense bf16 ~2.25 PFLOP/s, up from ~1 | datasheet |
| FA4 (§8) | paper March 2026, up to 1605 TFLOP/s bf16 on B200 (71%), 1.3× cuDNN 9.13, 2.7× Triton, 20–30× faster compiles with CuTe-DSL | FA4 paper. The primer did not fetch it on this date. |
| FA4 decode (§8) | initially slower than FA2 on B200, until split-KV came to FA4 | Project history. Examine the current release notes. |
| FlexAttention (§9) | backed by FA4 on Blackwell | PyTorch release in use |
