# roofline-core — predict what the hardware should do, with arithmetic

After this core you can compute, from a spec sheet and a model config, whether an LLM step is compute- or
memory-bound, what a collective costs, how long a cold start takes, how often a big job fails and what a token
costs. It is seven small standard-library modules — a dated accelerator catalogue, the roofline, the FLOPs and bytes
of an LLM step, the α-β cost of a collective, cold start, failure rates and the cost of a token — plus four fill-in
notebooks.

**Tier T0** (laptop, Colab CPU or CI; no GPU, no network; free). This is the *minimal* core of the topic. The
concepts are in [`../PRIMER.md`](../PRIMER.md) — every computed number it quotes comes from here and is pinned by
`tests/test_primer_numbers.py`. The *detailed* lab, [`../gpu-bench-lab/`](../gpu-bench-lab/), measures the same
quantities on the hardware you have.

## Start here

1. Read [`../PRIMER.md`](../PRIMER.md) §1–§2 (spec sheets and the roofline).
2. Run the tests (below): 58 tests, well under a second.
3. Open [`notebooks/01_spec_sheets_and_the_roofline.ipynb`](notebooks/01_spec_sheets_and_the_roofline.ipynb); each
   exercise's check cell prints ✅ when you are right.

## Run it

```bash
cd roofline-core
python3 -m pip install -r requirements.txt   # pytest only: enough for the tests
python3 -m pytest -q                          # 58 tests, well under a second
python3 -m pip install -r requirements-notebooks.txt   # JupyterLab (~250 MB), to do the notebooks locally
python3 -m jupyterlab notebooks               # do the exercises
```

`make setup test` and `make setup-notebooks lab` do the same. On Colab you need neither file: the notebooks' first
cell clones the repo and installs the library, and Colab already has Jupyter.

The library itself needs nothing installed:

```python
from roofline import fabric, llm, specs

h100 = specs.get("h100-sxm")
m = llm.PRESETS["llama-3.1-8b"]

llm.prefill(m, h100, 2048).bound              # 'compute'  (intensity ~1,941 FLOP/B, ridge 295)
step = llm.decode(m, h100, batch=1, context=1024)
step.bound, round(step.time * 1e3, 2)         # ('memory', 4.52)
llm.decode_intensity_limit(m, 4096)           # ~32 FLOP/B: KV reads cap the step's average, whatever the batch
llm.gemm_crossover_batch(m, h100)             # 296: per kernel, the weight GEMMs still reach the ridge
m70, nv, ib = llm.PRESETS["llama-3.1-70b"], fabric.LINKS["nvlink4"], fabric.LINKS["ib-ndr"]
fabric.tp_comm_time(m70, 4096, 8, nv)                       # 0.046 s of all-reduce per 4K prefill, TP=8 in a node
fabric.tp_comm_time_across_nodes(m70, 4096, 8, 2, nv, ib)   # 0.075 s at TP=16 over two nodes with rails
```

## What you get: the library (seven files)

| File | What it teaches |
|------|-----------------|
| `roofline/specs.py` | spec-sheet literacy: a dated (Sep 2026, verify) catalogue of 16 accelerators with **dense** peaks, per-direction links, `peak_from_clock`, `from_sparse` |
| `roofline/roofline.py` | `attainable = min(peak, I × BW)`, the ridge, kernel byte counts (elementwise, reduction, GEMM), tiling and fusion traffic (with extra inputs such as a residual), a text roofline chart |
| `roofline/llm.py` | one engine step from a model config: prefill vs decode FLOPs and bytes, the step as one kernel and split per kernel (weight GEMMs vs attention), KV reads that cap decode, crossover batches, memory-bound batch limits, quantization schemes, MoE experts touched |
| `roofline/fabric.py` | the link ladder, α-β, ring and recursive-doubling all-reduce, algbw/busbw, tensor-parallel cost per step in a node and across nodes, rails and hierarchical all-reduce, leaf-spine sizing, oversubscription, bisection, staged vs GPUDirect copies, `nvidia-smi topo -m` codes |
| `roofline/storage.py` | checkpoint bytes, tier bandwidths (assumptions), parallel and streamed loading, a cold-start breakdown |
| `roofline/reliability.py` | failure rates add, cluster MTBF from the Llama 3 data, Young/Daly checkpoint interval and waste, replicas as failure domains |
| `roofline/cost.py` | $/GPU-hr → $/M tokens, utilisation, rent vs own break-even |

Read them in that order. Each module opens with a docstring stating the one idea it teaches.

## The notebooks

Each opens with *The one-minute version*, works examples, then has 5–6 exercises (`# YOUR CODE HERE`)
each followed by a check cell that prints ✅, and ends with *In a design review* (a two-minute
explanation and drill questions). Solutions are in `solutions/`.

1. **`01_spec_sheets_and_the_roofline`** — read a datasheet (dense vs sparse, per-direction links, bits vs
   bytes), build the roofline, find when a GEMM becomes compute-bound, and why a kernel that saturates a T4
   can starve an H100. Primer §1–2.
2. **`02_llm_inference_on_the_roofline`** — prefill vs decode, the batch sweep, the KV ceiling on decode
   intensity, the closed-form crossover batch, the step per kernel (GEMMs vs attention), predicting a
   quantization speedup from bytes, picking a batch under an ITL SLO, MoE expert streaming, picking a GEMM
   tile that fits shared memory and fusing an elementwise chain. Primer §3–4.
3. **`03_fabrics_and_collective_cost`** — α-β, simulating a ring all-reduce, reading a busbw sweep, choosing
   a TP degree against an ITL SLO in or across nodes, sizing a two-tier fabric, reading `nvidia-smi topo -m`.
   Primer §5.
4. **`04_loading_reliability_and_cost`** — streamed loading, a cold-start budget, the Young/Daly interval,
   spares and failure-domain size, $/M tokens, rent vs own. Primer §6–8.

## Regenerating notebooks

`notebooks/` and `solutions/` are generated from `notebooks_src/*.py` (percent format with
`### BEGIN SOLUTION` blocks). Edit the sources, then:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
```

On Colab, each notebook's first cell clones the repository and installs this package; locally it finds
`roofline/` by walking up from the notebook's directory. Charts use `matplotlib` if it is installed
(`pip install -e ".[plot]"`) and fall back to text otherwise.

## Caveats: what the numbers are — and are not

Every time here is a **roofline bound**: ideal overlap, compulsory traffic, peak clocks. Real kernels and
engines land below it; the gap is what `gpu-bench-lab` and layer 04's `vllm-serving-lab` measure. Product
facts are a dated snapshot marked (verify); link latencies (α) and storage bandwidths are labelled
assumptions. MIT licensed.
