# kernel-core — the KV cache, paged attention and FlashAttention in numpy, tested on a laptop

After this you can show, with code that runs on any CPU, that a KV cache changes the cost of decoding and not its
output, that a block table with refcounts and copy-on-write changes where KV bytes live and not the attention
result, and that tiled attention with an online softmax is exact; `kerncore` also recomputes every number in the
three kernel primers of this layer.

## Start here

1. Read the [KV cache primer](../kv-cache/kv-cache-primer.md) §2–§4 (30 min).
2. `python3 -m pip install -e '.[dev]' && python3 -m pytest -q` — 51 tests in about a second, numpy only,
   including "cached decode == recomputing the prefix", "paged == contiguous" and "tiled == exact".
3. Open [`../kv-cache/01_kv_cache_worked.ipynb`](../kv-cache/01_kv_cache_worked.ipynb), then fill in the blanks of
   [`../kv-cache/02_kv_cache_practice.ipynb`](../kv-cache/02_kv_cache_practice.ipynb).

## What you get

*Tier T0 = laptop or Colab CPU, free: everything here runs with no GPU, no PyTorch and no network.* Together with
the three primers this is the repo curriculum's module 04.0 (about 5 hours).

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`kerncore/kv.py`](kerncore/kv.py) | size a KV cache per token, per layer and per request for MHA, GQA and MQA in fp16 or fp8 (binary units, GB in brackets); say how many sessions fit next to the weights; run a tiny decoder whose cached decode gives the same tokens and logits as recomputing, and count what each step costs | 30 min to read | T0 |
| [`kerncore/paged.py`](kerncore/paged.py) | allocate KV in blocks on demand, share them with refcounts, copy on write, and attend through a block table, gathered or one block at a time with the online softmax; flip a switch to see the two classic bugs (no running max, a copy-on-write that never releases) fail | 30 min | T0 |
| [`kerncore/flash.py`](kerncore/flash.py) | run the FlashAttention-2 forward schedule tile by tile with the online softmax and log-sum-exp, skip tiles above the causal diagonal, and see why a forward key loop needs no `-inf` guard while a backward one does; check tiles and bytes against the deep dive's `fa_calculators.py` | 45 min | T0 |
| [`../kv-cache/`](../kv-cache/kv-cache-primer.md) notebooks | the worked notebook (the cache in numpy, `IDENTICAL: True`, the cost curves, the sizes) and the practice notebook (four blanks, each with a check that fails the usual wrong answers) | ~1.5 h | T0 |

The paged-attention and flash-attention practice notebooks in [`../paged-attention/`](../paged-attention/paged_attention_practice.ipynb)
and [`../flash-attention/`](../flash-attention/flash_attention_practice.ipynb) are numpy already; `kerncore.paged` and
`kerncore.flash` are tested versions of their answer keys.

## Run it

```bash
cd 04-inference-engine/kernel-core
python3 -m pip install -e '.[dev]'    # numpy; plus pytest, matplotlib and Jupyter for the tests and notebooks
python3 -m pytest -q                  # 51 tests, ~1 s, offline
make check                            # the tests, then 01 runs clean and 02's blank stops at its first blank (~10 s)
python3 -m jupyterlab ../kv-cache     # do the notebooks
```

Install it editable (`-e`): `kerncore` imports [`fa_calculators.py`](../flash-attention/fa_calculators.py) and
[`paged_attention_minimal.py`](../paged-attention/paged_attention_minimal.py) from the checkout rather than copying
them. The library itself needs only numpy:

```python
from kerncore import kv, paged, flash

per_tok = kv.kv_bytes_per_token(32, 8, 128)               # Llama 3 8B, fp16
print(kv.fmt_bytes(per_tok), kv.fmt_bytes(kv.kv_cache_bytes(32, 8, 128, 128_000)))
# 128 KiB 15.62 GiB (16.78 GB)

model = kv.TinyDecoder(kv.random_params(seed=0))
print(kv.compare(model, [1, 2, 3, 4], 16)["identical"])    # True: same ids, logits within 1e-9

import numpy as np
Q = K = V = np.random.default_rng(0).standard_normal((512, 8))
O, lse, stats = flash.flash_attention(Q, K, V, 128, 64, causal=True)
print(stats["visited"], stats["masked"], stats["grid"])      # 20 8 32, as fa_calculators.causal_tiles predicts
```

The kv-cache notebooks are hand-written `.ipynb` files kept in [`../kv-cache/`](../kv-cache/) (not generated from
percent-format sources); their first cell is the repo's Colab setup cell, and the next one puts this folder on
`sys.path`, so they need no install on Colab. `make notebooks` re-executes the worked notebook and saves its
outputs; `tests/test_notebooks.py` checks that the saved numbers are the ones `kerncore` computes, that the
practice notebook is committed blank, and that torch appears only in 01's optional, guarded comparison cell.

## How it fits

Read it next to the three primers of this layer: the [KV cache primer](../kv-cache/kv-cache-primer.md) (every size
there is pinned by `tests/test_primer_numbers.py`), the [paged-attention primer](../paged-attention/paged-attention-primer.md)
(`kerncore.paged` is `paged_attention_minimal.py` as objects, cross-checked in `tests/test_paged.py`) and the
[FlashAttention primer](../flash-attention/flash-attention-primer.md) with its
[deep dive](../flash-attention/flash-attention-deep-dive.md) (§4.4 tile skipping and §11.2's forward-loop point are
tests in `tests/test_flash.py`). It builds on [`00-foundations/transformers`](../../00-foundations/transformers/)
(attention and decoding). Next is [`serving-engine/mini-engine-core`](../serving-engine/mini-engine-core/README.md),
where the same pieces run inside an engine: a model that reads K/V through the block tables of a flat batch of many
requests, a block manager with prefix caching, and a scheduler that decides who gets the blocks.

## Caveats

- Everything is float64 numpy. "IDENTICAL" means the same token ids and every step's logits within 1e-9 (the two
  paths multiply matrices of different shapes, so the last bits can differ); tiled and paged attention match the
  exact result to 1e-12.
- Costs are counted, not measured: multiply-adds in the tiny decoder, and the bytes a 16-bit kernel with the FA2
  schedule would move under `fa_calculators`' no-L2 model. The wall-clock line in the worked notebook is whatever
  CPU ran it.
- `kv.sessions_per_gpu` is an upper bound (no activations, allocator reserve or fragmentation). Model shapes and GPU
  figures come from the primers' verify lists, dated 2026-09-26 (verify).
- The paged pool and the tiled kernel are single-head; GQA appears in `kv.TinyDecoder` and the sizing only. MIT
  licensed.
