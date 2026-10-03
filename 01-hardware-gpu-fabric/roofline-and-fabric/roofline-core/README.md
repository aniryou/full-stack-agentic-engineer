# roofline-core — predict what the hardware should do, with arithmetic

After this core, you can calculate these values from a spec sheet and a model config:

- if an LLM step is compute-bound or memory-bound,
- what a collective costs,
- how long a cold start takes,
- how often a large job fails,
- what a token costs.

The core has seven small standard-library modules and four fill-in notebooks. The modules cover these topics:

- a dated accelerator catalogue,
- the roofline,
- the FLOPs and bytes of an LLM step,
- the α-β cost of a collective,
- cold start,
- failure rates,
- the cost of a token.

**Tier T0** (laptop, Colab CPU or CI). It needs no GPU and no network, and it is free. This is the *minimal* core of
the topic. [`../PRIMER.md`](../PRIMER.md) explains the concepts. Each computed number that the primer quotes comes
from this core, and `tests/test_primer_numbers.py` pins it. The *detailed* lab,
[`../gpu-bench-lab/`](../gpu-bench-lab/), measures the same quantities on the hardware that you have.

## Start here

1. Read [`../PRIMER.md`](../PRIMER.md) §1–§2 (spec sheets and the roofline).
2. Run the tests (see "Run it"): 66 tests, ~30 s.
3. Open [`notebooks/01_spec_sheets_and_the_roofline.ipynb`](notebooks/01_spec_sheets_and_the_roofline.ipynb). The
   check cell of each exercise prints ✅ when your answer is correct.

## Run it

```bash
cd roofline-core
python3 -m pip install -r requirements.txt   # pytest + the notebook tooling (~15 MB): enough for the tests
python3 -m pytest -q                          # 66 tests, ~30 s
python3 -m pip install -r requirements-notebooks.txt   # JupyterLab (~250 MB), to do the notebooks locally
python3 -m jupyterlab notebooks               # do the exercises
```

`make setup test` and `make setup-notebooks lab` do the same steps. `make notebooks` and `make check` also need
`make setup-notebooks`. On Colab, you need neither file. The first cell of each notebook clones the repo and
installs the library, and Colab already has Jupyter.

The library itself needs no installed packages:

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
| `roofline/specs.py` | How to read a spec sheet: a dated (Sep 2026, verify) catalogue of 16 accelerators with **dense** peaks, per-direction links, `peak_from_clock`, `from_sparse` |
| `roofline/roofline.py` | `attainable = min(peak, I × BW)`, the ridge, kernel byte counts (elementwise, reduction, GEMM), tiling and fusion traffic (with extra inputs such as a residual), a text roofline chart |
| `roofline/llm.py` | One engine step from a model config. Prefill against decode FLOPs and bytes. The step as one kernel and divided per kernel (weight GEMMs against attention). KV reads that cap decode, crossover batches, memory-bound batch limits, quantization schemes, MoE experts touched. |
| `roofline/fabric.py` | The link ladder, α-β, ring and recursive-doubling all-reduce, algbw/busbw. Tensor-parallel cost per step in a node and across nodes. Rails and hierarchical all-reduce, the size of a leaf-spine fabric, oversubscription, bisection. Staged against GPUDirect copies, `nvidia-smi topo -m` codes. |
| `roofline/storage.py` | Checkpoint bytes, tier bandwidths (assumptions), parallel and streamed loading, a cold-start breakdown |
| `roofline/reliability.py` | How failure rates add, cluster MTBF from the Llama 3 data, the Young/Daly checkpoint interval and its waste, replicas as failure domains |
| `roofline/cost.py` | From $/GPU-hr to $/M tokens, utilisation, the break-even of rent against own |

Read them in that order. Each module starts with a docstring. The docstring states the one idea that the module
teaches.

## The notebooks

Each notebook starts with *The one-minute version* and shows worked examples. Then it has 5–6 exercises
(`# YOUR CODE HERE`). After each exercise, a check cell prints ✅. The notebook ends with *In a design review* (a
two-minute explanation and drill questions). The solutions are in `solutions/`.

1. **`01_spec_sheets_and_the_roofline`**: you read a datasheet (dense against sparse, per-direction links, bits
   against bytes). You build the roofline and find when a GEMM becomes compute-bound. You also see why a kernel
   that saturates a T4 can fail to keep an H100 busy. Primer §1–2.
2. **`02_llm_inference_on_the_roofline`**: you examine prefill against decode, the batch sweep, the KV ceiling on
   decode intensity and the closed-form crossover batch. You examine the step per kernel (GEMMs against attention)
   and predict a quantization speedup from bytes. You select a batch under an ITL SLO and examine MoE expert streaming.
   Last, you select a GEMM tile that fits shared memory and fuse an elementwise chain. Primer §3–4.
3. **`03_fabrics_and_collective_cost`**: you examine α-β. You simulate a ring all-reduce and read a busbw sweep. You
   select a TP degree against an ITL SLO in a node or across nodes. Then you calculate the size of a two-tier
   fabric and read `nvidia-smi topo -m`. Primer §5.
4. **`04_loading_reliability_and_cost`**: you examine streamed loading, a cold-start budget and the Young/Daly
   interval. You also examine spares and failure-domain size, $/M tokens, and rent against own. Primer §6–8.

## Regenerating notebooks

The builder generates `notebooks/` and `solutions/` from `notebooks_src/*.py` (percent format with
`### BEGIN SOLUTION` blocks). The build and run tools need the notebook requirements, not only the test
requirements (`requirements.txt` is pytest only). Thus install the notebook requirements first. Edit the sources.
Then run these commands:

```bash
python3 -m pip install -r requirements-notebooks.txt    # nbformat, nbclient, ipykernel, JupyterLab (or make setup-notebooks)
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
```

On Colab, the first cell of each notebook clones the repository and installs this package. Locally, the notebook
walks up from its directory until it finds `roofline/`. Charts use `matplotlib` when the package is present
(`pip install -e ".[plot]"`). If the package is not present, the code draws the charts as text.

## Caveats: what the numbers are — and are not

Each time in this core is a **roofline bound**. The bound assumes ideal overlap, compulsory traffic and peak clocks.
Real kernels and engines do not reach it. `gpu-bench-lab` and the `vllm-serving-lab` of layer 04 measure the gap.
The product facts are a dated snapshot with the tag (verify). The link latencies (α) and the storage bandwidths are
assumptions, and they have that label.

The core has the MIT license.
