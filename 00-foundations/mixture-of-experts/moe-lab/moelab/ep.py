"""ep.py — expert parallelism versus tensor parallelism on two GPUs: predict it, run it, read it.

One idea: with two GPUs there are three ways vLLM can lay out an MoE, and they differ in what each
GPU streams, how evenly, and what crosses the link every layer:

    layout   vLLM flags                                              attention      experts per GPU     per MoE layer
    tp       --tensor-parallel-size 2                                TP-sharded     half of EVERY one   all-reduce (after attn too)
    tp_ep    --tensor-parallel-size 2 --enable-expert-parallel       TP-sharded     ALL of half of them all-reduce (after attn too)
    dp_ep    --data-parallel-size 2 --enable-expert-parallel         replicated,    ALL of half of them all-gather + reduce-scatter
                                                                     own requests                       (no attention collective)

EP size is not a flag: it is TP x DP when ``--enable-expert-parallel`` is set; without it the
experts are tensor-parallel over TP x DP GPUs. With DP = 1 the tokens are already on both GPUs
after attention, so vLLM runs no all-to-all at all — each GPU applies its own experts and one
all-reduce combines them (``use_all2all_kernels`` requires DP > 1, vllm/model_executor/layers/
fused_moe/config.py, v0.30.0). With DP = 2 the default ``--all2all-backend`` is
``allgather_reducescatter``. The "tokens x k x hidden x bytes" dispatch and combine of layer 02's
PRIMER §5.6 is what dedicated all-to-all kernels (DeepEP, SM90+ with NVLink/RDMA — never on T4/L4)
move; on two PCIe GPUs the collectives above are what you get.

Whatever the layout, a step ends when the slowest GPU finishes: under EP the GPU holding the hot
experts streams and computes more (imbalance), under TP both halves are equal by construction.
The step model here is the roofline plus stated efficiencies (**simulated**, see ``stream.SimParams``)
and an alpha-beta link (``Link``: assumed until you fit your own with layer 02's cuda-nccl-lab).
``parse_bench_serve`` reads what ``vllm bench serve`` prints, so a real run replaces the prediction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .configs import GPU, Model
from .hooks import placement
from .stream import SimParams, sample_topk, zipf_popularity

LAYOUTS = ("tp", "tp_ep", "dp_ep")


def vllm_flags(layout: str, gpus: int = 2) -> list[str]:
    """The ``vllm serve`` flags for a layout on ``gpus`` GPUs (vLLM v0.30.0 names)."""
    return {"tp": ["--tensor-parallel-size", str(gpus)],
            "tp_ep": ["--tensor-parallel-size", str(gpus), "--enable-expert-parallel"],
            "dp_ep": ["--data-parallel-size", str(gpus), "--enable-expert-parallel"]}[layout]


def ep_size(tp: int, dp: int, enable_expert_parallel: bool) -> int:
    """vLLM: EP_SIZE = TP x DP with --enable-expert-parallel; otherwise experts are TP-sharded (EP 1)."""
    return tp * dp if enable_expert_parallel else 1


# ---- bytes on the wire -------------------------------------------------------------------------------
def a2a_bytes(tokens: int, top_k: int, hidden: int, elem_bytes: float = 2, scale_block: int = 0,
              scale_bytes: float = 4) -> float:
    """One all-to-all direction, per GPU, upper bound (every assignment remote): tokens x k x hidden x
    bytes (layer 02 PRIMER §5.6), plus one ``scale_bytes`` scale per ``scale_block`` channels for FP8
    dispatch (DeepEP ships 4-byte scales per 128 channels). Uniform routing over ``ep`` ranks sends
    (ep - 1)/ep of it off the GPU."""
    per = hidden * elem_bytes + (hidden / scale_block * scale_bytes if scale_block else 0)
    return tokens * top_k * per


@dataclass(frozen=True)
class Link:
    """alpha-beta parameters of the link between the GPUs: t(S) = alpha + S / bw per step."""
    name: str
    alpha_s: float
    bw_gbs: float
    note: str = ""


LINKS: dict[str, Link] = {
    # Layer 02's worked example (cuda-nccl-core notebook 03): 8 GPUs, alpha = 2 us, B = 450 GB/s.
    "nvlink-8gpu": Link("NVLink/NVSwitch (layer 02's example)", 2e-6, 450.0, "layer 02 PRIMER §5.6 parameters"),
    # Two GPUs on PCIe without NVLink: NCCL uses P2P or shared host memory. ASSUMED values (verify):
    # fit your own with 02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab notebook 04 and pass Link(...).
    "pcie-2xT4": Link("2 x T4, PCIe Gen3 (Kaggle 'GPU T4 x2')", 20e-6, 8.0, "assumed, not measured"),
    "pcie-2xL4": Link("2 x L4, PCIe Gen4 (g2-standard-24)", 15e-6, 12.0, "assumed, not measured"),
}


def collective_time(op: str, size: float, p: int, link: Link, algo: str = "ring") -> float:
    """alpha-beta cost of one collective over a ``size``-byte buffer (layer 02's ``cost_terms``):
    ring all-reduce 2(p-1) alpha + 2(p-1)/p S/B; ring all-gather / reduce-scatter and pairwise
    all-to-all (p-1) alpha + (p-1)/p S/B; direct all-to-all alpha + (p-1)/p S/B."""
    if p < 2:
        return 0.0
    f = (p - 1) / p
    a, c = {("all_reduce", "ring"): (2 * (p - 1), 2 * f), ("all_gather", "ring"): (p - 1, f),
            ("reduce_scatter", "ring"): (p - 1, f), ("all_to_all", "pairwise"): (p - 1, f),
            ("all_to_all", "direct"): (1, f)}[(op, algo)]
    return a * link.alpha_s + c * size / (link.bw_gbs * 1e9)


def moe_layer_comm(layout: str, tokens: int, hidden: int, link: Link, p: int = 2, elem_bytes: float = 2) -> float:
    """Seconds of communication per transformer layer for ``tokens`` tokens in the step."""
    s = tokens * hidden * elem_bytes
    if layout in ("tp", "tp_ep"):          # after attention's o_proj and after the MoE
        return 2 * collective_time("all_reduce", s, p, link)
    if layout == "dp_ep":                  # allgather_reducescatter: gather tokens, scatter results back
        return collective_time("all_gather", s, p, link) + collective_time("reduce_scatter", s, p, link)
    raise ValueError(layout)


# ---- what each GPU holds -------------------------------------------------------------------------------
def per_gpu_weight_bytes(model: Model, layout: str, p: int = 2, weight_bytes: float = 2) -> float:
    """Weights one GPU holds. TP shards attention, shared expert and the vocab (embedding and LM
    head); EP gives each GPU E/p whole experts; DP replicates everything except the experts."""
    L = model.layers
    attn, shared, router = L * model.attn_params(), L * model.shared_params(), L * model.router_params()
    experts, emb = model.expert_param_total(), model.embedding_params()
    if layout in ("tp", "tp_ep"):
        per = (attn + shared + emb + experts) / p + router
    elif layout == "dp_ep":
        per = attn + shared + emb + router + experts / p
    else:
        raise ValueError(layout)
    return per * weight_bytes


# ---- one decode step on p GPUs --------------------------------------------------------------------------
@dataclass
class EPStep:
    layout: str
    batch: int
    rank_s: list = field(default_factory=list)       # compute/memory time of each GPU (before comm)
    rank_bytes: list = field(default_factory=list)
    comm_s: float = 0.0
    overhead_s: float = 0.0

    @property
    def time(self) -> float:
        return self.overhead_s + max(self.rank_s) + self.comm_s

    @property
    def imbalance(self) -> float:
        """Slowest GPU over the mean GPU (1.0 = balanced)."""
        return max(self.rank_s) / (sum(self.rank_s) / len(self.rank_s))


def routes(model: Model, batch: int, s: float = 0.0, seed: int = 0) -> np.ndarray:
    """``[batch, layers, k]`` simulated routing (Zipf(s) popularity per layer; s = 0 is uniform)."""
    rng = np.random.default_rng(seed)
    return np.stack([sample_topk(zipf_popularity(model.n_experts, s, seed=seed + l), model.top_k, batch, rng)
                     for l in range(model.layers)], axis=1)


def ep_decode_step(model: Model, gpu: GPU, layout: str, batch: int, context: int, link: Link, *,
                   ids: np.ndarray | None = None, p: int = 2, weight_bytes: float = 2, kv_bytes: float = 2,
                   params: SimParams = SimParams(), strategy: str = "linear") -> EPStep:
    """Per-GPU roofline time for one decode step of ``batch`` sequences (**simulated**), then comm.

    ``ids`` ``[batch, layers, k]`` are the routing decisions (default: uniform, ``routes()``); pass a
    captured trace to see what real skew does to the EP ranks."""
    ids = routes(model, batch) if ids is None else ids[:batch]
    L, d = model.layers, model.d_model
    where = placement(model.n_experts, p, strategy)
    e_bytes, e_flops = model.expert_params() * weight_bytes, 2 * model.expert_params()
    kv_tok = model.kv_bytes_per_token(kv_bytes) * (context + 1)
    attn_flops_tok = 2 * model.attn_params() + 4 * model.n_heads * model.head_dim * (context + 1)
    head = model.vocab * d
    rank_bytes, rank_flops = np.zeros(p), np.zeros(p)
    for l in range(L):
        counts = np.bincount(ids[:, l, :].ravel(), minlength=model.n_experts)
        touched = counts > 0
        if layout == "tp":                                   # every GPU: half of each touched expert
            rank_bytes += touched.sum() * e_bytes / p
            rank_flops += counts.sum() * e_flops / p
        else:                                                # whole experts on their home GPU
            rank_bytes += np.bincount(where, weights=touched * e_bytes, minlength=p)
            rank_flops += np.bincount(where, weights=counts * e_flops, minlength=p)
    shared_router = L * (model.shared_params() + model.router_params())
    if layout in ("tp", "tp_ep"):
        rank_bytes += (L * model.attn_params() + head) * weight_bytes / p + batch * kv_tok / p
        rank_bytes += L * model.router_params() * weight_bytes + L * model.shared_params() * weight_bytes / p
        rank_flops += batch * (L * attn_flops_tok + 2 * shared_router + 2 * head) / p
    else:                                                    # dp_ep: each GPU its own batch/p sequences
        mine = np.array([len(c) for c in np.array_split(np.arange(batch), p)])
        rank_bytes += (L * model.attn_params() + shared_router + head) * weight_bytes + mine * kv_tok
        rank_flops += mine * (L * attn_flops_tok + 2 * shared_router + 2 * head)
    t_mem = rank_bytes / (gpu.mem_bw_gbs * 1e9 * params.mem_eff)
    t_cmp = rank_flops / (gpu.tflops_16 * 1e12 * params.compute_eff)
    comm = L * moe_layer_comm(layout, batch, d, link, p)
    return EPStep(layout, batch, np.maximum(t_mem, t_cmp).tolist(), rank_bytes.tolist(), comm, params.overhead_s)


def compare_layouts(model: Model, gpu: GPU, link: Link, batches=(1, 8, 32, 128), context: int = 512,
                    s: float = 0.0, **kw) -> list[dict]:
    """ITL (ms, **simulated**) of each layout at each batch, plus the EP imbalance."""
    rows = []
    for b in batches:
        ids = routes(model, b, s=s)
        row = {"batch": b}
        for lay in LAYOUTS:
            st = ep_decode_step(model, gpu, lay, b, context, link, ids=ids, **kw)
            row[lay] = st.time * 1e3
            row[f"{lay}_imbalance"] = st.imbalance
            row[f"{lay}_comm_ms"] = st.comm_s * 1e3
        rows.append(row)
    return rows


def format_rows(rows: list[dict]) -> str:
    out = ["batch   tp (ms)  tp_ep (ms)  dp_ep (ms)   EP imbalance   comm tp/dp_ep (ms)   [simulated]"]
    for r in rows:
        out.append(f"{r['batch']:5d} {r['tp']:9.2f} {r['tp_ep']:11.2f} {r['dp_ep']:11.2f} {r['tp_ep_imbalance']:12.2f}"
                   f" {r['tp_comm_ms']:10.2f}/{r['dp_ep_comm_ms']:.2f}")
    return "\n".join(out)


# ---- vllm bench serve: the command and its printed result ---------------------------------------------
def bench_command(model: str, concurrency: int, *, base_url: str = "http://127.0.0.1:8000", input_len: int = 256,
                  output_len: int = 128, num_prompts: int | None = None) -> list[str]:
    """A closed-loop ``vllm bench serve`` run (flags of vLLM v0.30.0's benchmarks/serve.py and
    datasets.py): random prompts of fixed length, ``--ignore-eos`` so every request decodes
    ``output_len`` tokens, ``--max-concurrency`` users."""
    return ["vllm", "bench", "serve", "--base-url", base_url, "--model", model, "--dataset-name", "random",
            "--random-input-len", str(input_len), "--random-output-len", str(output_len),
            "--num-prompts", str(num_prompts or 8 * concurrency), "--max-concurrency", str(concurrency),
            "--ignore-eos"]


_KEY = re.compile(r"[^a-z0-9]+")


def parse_bench_serve(text: str) -> dict:
    """Read the ``Serving Benchmark Result`` block that ``vllm bench serve`` prints — lines of the
    form ``"{label:<40} {value}"`` — into ``{"mean_itl_ms": 41.2, "output_token_throughput_tok_s": ...}``."""
    out, inside = {}, False
    for line in text.splitlines():
        if "Serving Benchmark Result" in line:
            inside = True
            continue
        if not inside:
            continue
        if line.startswith("=" * 50):
            break
        m = re.match(r"^(.+?:)\s+(-?[\d.]+)\s*$", line)
        if m:
            key = _KEY.sub("_", m.group(1).rstrip(":").lower()).strip("_")
            v = float(m.group(2))
            out[key] = int(v) if v.is_integer() and "." not in m.group(2) else v
    return out
