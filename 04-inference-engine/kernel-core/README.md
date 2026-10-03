# kernel-core — the KV cache, paged attention and FlashAttention in numpy, tested on a laptop

After this, you can show three things with code that runs on any CPU:

- A KV cache changes the cost of decode, but not its output.
- A block table with refcounts and copy-on-write changes where KV bytes live, but not the attention result.
- Tiled attention with an online softmax is exact.

`kerncore` also recomputes every number in the three kernel primers of this layer.

## Start here

1. Read the [KV cache primer](../kv-cache/kv-cache-primer.md) §2–§4 (30 min).
2. Run `python3 -m pip install -e '.[dev]' && python3 -m pytest -q`. The 51 tests run in about a second and
   need only numpy. They include "cached decode == recomputing the prefix", "paged == contiguous" and "tiled == exact".
3. Open [`../kv-cache/notebooks/01_kv_cache_worked.ipynb`](../kv-cache/notebooks/01_kv_cache_worked.ipynb). Then fill in the blanks of
   [`../kv-cache/notebooks/02_kv_cache_practice.ipynb`](../kv-cache/notebooks/02_kv_cache_practice.ipynb).

## What you get

*Tier T0 = laptop or Colab CPU, free. Everything here runs with no GPU, no PyTorch and no network.* Together with
the three primers, this core is module 04.0 of the repo curriculum (about 5 hours).

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`kerncore/kv.py`](kerncore/kv.py) | Size a KV cache per token, per layer and per request for MHA, GQA and MQA in fp16 or fp8 (binary units, GB in brackets). Say how many sessions fit next to the weights. Run a small decoder. Its cached decode gives the same tokens and logits as the path that recomputes the prefix. Count what each step costs. | 30 min to read | T0 |
| [`kerncore/paged.py`](kerncore/paged.py) | Allocate KV in blocks on demand. Share the blocks with refcounts and copy them on write. Compute attention through a block table, with the blocks gathered, or one block at a time with the online softmax. Set a switch and see the two classic bugs fail: no running max, and a copy-on-write that never releases. | 30 min | T0 |
| [`kerncore/flash.py`](kerncore/flash.py) | Run the FlashAttention-2 forward schedule tile by tile, with the online softmax and log-sum-exp. Skip the tiles above the causal diagonal. See why a forward key loop needs no `-inf` guard, but a backward key loop needs one. Compare the tiles and bytes with the deep dive's `fa_calculators.py`. | 45 min | T0 |
| [`../kv-cache/`](../kv-cache/kv-cache-primer.md) notebooks | The worked notebook: the cache in numpy, `IDENTICAL: True`, the cost curves, the sizes. The practice notebook: four blanks, each with a check that fails the usual incorrect answers. | ~1.5 h | T0 |

The paged-attention and flash-attention practice notebooks in [`../paged-attention/notebooks/`](../paged-attention/notebooks/paged_attention_practice.ipynb)
and [`../flash-attention/notebooks/`](../flash-attention/notebooks/flash_attention_practice.ipynb) already use numpy. `kerncore.paged` and
`kerncore.flash` are versions of their answer keys, with tests.

## Run it

```bash
cd 04-inference-engine/kernel-core
python3 -m pip install -e '.[dev]'    # numpy; plus pytest, matplotlib and Jupyter for the tests and notebooks
python3 -m pytest -q                  # 51 tests, ~1 s, offline
make check                            # the tests, then 01 runs clean and 02's blank stops at its first blank (~10 s)
python3 -m jupyterlab ../kv-cache     # do the notebooks
```

Install it editable (`-e`). `kerncore` imports [`fa_calculators.py`](../flash-attention/fa_calculators.py) and
[`paged_attention_minimal.py`](../paged-attention/paged_attention_minimal.py) from the checkout, and does not copy
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

The kv-cache notebooks are `.ipynb` files written by hand and kept in [`../kv-cache/`](../kv-cache/). No
percent-format source generates them. Their first cell is the Colab setup cell of the repo. The next cell puts this
folder on `sys.path`, thus they need no install on Colab.

`make notebooks` runs the worked notebook again and saves its outputs. `tests/test_notebooks.py` makes sure of three
things:

- The saved numbers are the numbers that `kerncore` computes.
- The committed practice notebook is blank.
- Torch appears only in the optional, guarded comparison cell of 01.

## How it fits

Read this core next to the three primers of this layer:

- The [KV cache primer](../kv-cache/kv-cache-primer.md). `tests/test_primer_numbers.py` pins every size in it.
- The [paged-attention primer](../paged-attention/paged-attention-primer.md). `kerncore.paged` is
  `paged_attention_minimal.py` as objects, and `tests/test_paged.py` compares the two.
- The [FlashAttention primer](../flash-attention/flash-attention-primer.md) with its
  [deep dive](../flash-attention/flash-attention-deep-dive.md). The tile skip of §4.4 and the forward-loop point of
  §11.2 are tests in `tests/test_flash.py`.

This core builds on [`00-foundations/transformers`](../../00-foundations/transformers/) (attention and decode). The
next step is [`serving-engine/mini-engine-core`](../serving-engine/mini-engine-core/README.md). There, the same
pieces run inside an engine:

- a model that reads K/V through the block tables of a flat batch of many requests,
- a block manager with prefix caching,
- a scheduler that decides which requests get the blocks.

## Caveats

- Everything is float64 numpy. "IDENTICAL" means the same token ids, and the logits of every step within 1e-9.
  The two paths multiply matrices of different shapes, thus the last bits can be different. Tiled and paged
  attention match the exact result to 1e-12.
- The core counts the costs. It does not measure them. It counts the multiply-adds in the small decoder. It also
  counts the bytes that a 16-bit kernel with the FA2 schedule moves in the no-L2 model of `fa_calculators`. The
  wall-clock line in the worked notebook comes from the CPU that ran it, whatever that CPU is.
- `kv.sessions_per_gpu` gives an upper bound. It does not count activations, the allocator reserve or
  fragmentation. The model shapes and GPU figures come from the verify lists of the primers, dated 2026-09-26
  (verify).
- The paged pool and the tiled kernel are single-head. GQA appears only in `kv.TinyDecoder` and in the size
  calculations. The licence is MIT.
