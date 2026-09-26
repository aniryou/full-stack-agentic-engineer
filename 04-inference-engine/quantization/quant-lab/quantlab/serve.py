"""serve.py — which quantization scheme a GPU can run, how vLLM runs it, and the flags to ask for it.

One idea: the same checkpoint takes different paths on different GPUs. A W8A8 FP8 checkpoint
multiplies in FP8 on an L4 or H100 (compute capability 8.9+), but on a T4 or A100 vLLM loads it
as *weight-only* FP8 through Marlin (a memory win, no FLOP win); an NVFP4 checkpoint is W4A4 only
on Blackwell (sm_100+), weight-only elsewhere; INT4 W4A16 runs Machete on an H100 and Marlin on
everything else from sm_75; INT8 W8A8 is refused on sm_100+. ``plan(scheme, gpu)`` encodes those
rules — read from vLLM's source at v0.30.0 / main (Sep 2026) and the topic's fact sheet, every one
``(verify)`` when you move the pin — and renders the ``vllm serve`` / ``docker run`` command, the
Cloud Run variables for the serving lab's Terraform and the container args for its GKE manifest.
Pre-quantized checkpoints need no ``--quantization`` flag (vLLM reads ``quantization_config``;
passing a different one is an error); a bf16 checkpoint is quantized at load time with the online
names (``fp8_per_tensor``: ``--quantization fp8`` on a bf16 checkpoint works in v0.30.0 but raises
on main). PRIMER §9 "Producing a checkpoint" and §10 "Choosing a scheme" explain the table.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

IMAGE = "vllm/vllm-openai:v0.30.0"


@dataclass(frozen=True)
class GPU:
    """Datasheet numbers, dense (no sparsity), all (verify). ``memory_gib`` is what the driver
    reports (the serving lab's ``sizing.GPUS``); TFLOP/s per precision from the roofline core's
    ``specs.py`` (the RTX 4090 is only in the serving lab's table, unverified GeForce rates)."""
    name: str
    cc: float                    # compute capability
    memory_gib: float
    mem_bw_gbs: float
    tflops: dict
    arch: str = ""

    @property
    def sm(self) -> int:
        return int(round(self.cc * 10))

    def peak(self, precision: str) -> float:
        """Dense FLOP/s for a precision; bf16 falls back to fp16 (a T4 has no bf16 units)."""
        t = self.tflops.get(precision) or (self.tflops.get("fp16") if precision == "bf16" else None)
        if not t:
            raise KeyError(f"{self.name} has no {precision} tensor-core path")
        return t * 1e12

    @property
    def has_bf16(self) -> bool:
        return "bf16" in self.tflops


GPUS = {
    "T4": GPU("T4", 7.5, 15.0, 320, {"fp16": 65, "int8": 130}, "Turing"),
    "A100-40GB": GPU("A100-40GB", 8.0, 40.0, 1555, {"bf16": 312, "fp16": 312, "int8": 624}, "Ampere"),
    "A100-80GB": GPU("A100-80GB", 8.0, 80.0, 2039, {"bf16": 312, "fp16": 312, "int8": 624}, "Ampere"),
    "L4": GPU("L4", 8.9, 22.49, 300, {"bf16": 121, "fp16": 121, "fp8": 242.5, "int8": 242.5}, "Ada"),
    "RTX4090": GPU("RTX4090", 8.9, 23.99, 1008, {"bf16": 165, "fp16": 165, "fp8": 330, "int8": 330}, "Ada"),
    "H100-80GB": GPU("H100-80GB", 9.0, 79.65, 3350, {"bf16": 989.4, "fp16": 989.4, "fp8": 1978.9, "int8": 1978.9},
                     "Hopper"),
    "B200": GPU("B200", 10.0, 179.0, 8000, {"bf16": 2250, "fp16": 2250, "fp8": 4500, "fp4": 9000, "int8": 4500},
                "Blackwell"),
    "RTXPRO6000": GPU("RTXPRO6000", 12.0, 95.0, 1600, {"bf16": 500, "fp16": 500, "fp8": 1000, "fp4": 2000},
                      "Blackwell"),
}


def gpu(name) -> GPU:
    if isinstance(name, GPU):
        return name
    key = str(name).upper().replace(" ", "").replace("_", "-")
    for k, v in GPUS.items():
        if k.upper() == key or k.upper().split("-")[0] == key:
            return v
    raise KeyError(f"unknown GPU {name!r}; known: {', '.join(GPUS)}")


# ---------------------------------------------------------------------------------------------
# Schemes as a serving decision
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Scheme:
    key: str
    title: str
    weight_bits: float           # stored bits per quantized weight, scales included
    activation: str              # "16" | "fp8" | "int8" | "fp4": what the tensor cores multiply
    checkpoint: str              # how you get one
    t1_model: str                # a model id to try at T1 (verify that it exists)
    flags: tuple = ()            # vLLM flags beyond the model id


SCHEMES = {
    "bf16": Scheme("bf16", "BF16 / FP16 baseline", 16, "16", "the original checkpoint",
                   "Qwen/Qwen2.5-1.5B-Instruct"),
    "fp8": Scheme("fp8", "FP8 W8A8 (per-channel weights, dynamic per-token activations)", 8, "fp8",
                  "llm-compressor FP8_DYNAMIC (no data) -> compressed-tensors; auto-detected",
                  "./Qwen2.5-1.5B-Instruct-FP8_DYNAMIC"),
    "fp8-online": Scheme("fp8-online", "FP8 quantized at load time (per-tensor)", 8, "fp8",
                         "none: a bf16 checkpoint + --quantization fp8_per_tensor",
                         "Qwen/Qwen2.5-1.5B-Instruct", ("--quantization", "fp8_per_tensor")),
    "w4a16": Scheme("w4a16", "INT4 weight-only, groups of 128 (GPTQ / AWQ / compressed-tensors W4A16)", 4 + 16 / 128,
                    "16", "llm-compressor GPTQModifier(scheme=\"W4A16\") or a published AWQ/GPTQ checkpoint",
                    "Qwen/Qwen2.5-1.5B-Instruct-AWQ"),
    "w8a8-int8": Scheme("w8a8-int8", "INT8 W8A8 (SmoothQuant + GPTQ, dynamic per-token activations)", 8, "int8",
                        "llm-compressor [SmoothQuantModifier, GPTQModifier(scheme=\"W8A8\")]",
                        "./Qwen2.5-1.5B-Instruct-W8A8"),
    "nvfp4": Scheme("nvfp4", "NVFP4 W4A4 (E2M1, E4M3 scale per 16, FP32 per tensor)", 4.5, "fp4",
                    "llm-compressor QuantizationModifier(scheme=\"NVFP4\"), 20 calibration samples",
                    "./Qwen2.5-1.5B-Instruct-NVFP4"),
}


@dataclass
class ServePlan:
    scheme: str
    gpu: str
    supported: bool
    model: str
    flags: list
    compute: str                 # what the GEMMs actually do on this GPU
    kernel: str                  # the vLLM linear kernel it should log (verify: "Selected ... for ..." / "Using ... for ...")
    weight_bits: float
    notes: list = field(default_factory=list)

    def command(self) -> str:
        return " ".join(["vllm serve", self.model, *self.flags])

    def docker(self, image: str = IMAGE, port: int = 8000) -> str:
        return (f"docker run --rm --gpus all --ipc=host -p {port}:8000 -v $HOME/.cache/huggingface:/root/.cache/huggingface "
                f"{image} {self.model} {' '.join(self.flags)}".strip())

    def table_row(self) -> str:
        return (f"| {self.gpu} | {self.scheme} | {'yes' if self.supported else 'no'} | {self.compute} | "
                f"{self.kernel} | `{' '.join(self.flags) or '-'}` |")


def plan(scheme: str, gpu_name, *, model: str | None = None, kv_cache_dtype: str = "auto",
         max_model_len: int = 4096, gpu_memory_utilization: float = 0.92) -> ServePlan:
    """The serving decision for ``scheme`` on ``gpu_name``: supported or not, the compute path,
    the kernel vLLM should pick, the flags. Rules from vLLM v0.30.0 / main source (verify)."""
    s, g = SCHEMES[scheme], gpu(gpu_name)
    sm, notes = g.sm, []
    flags = list(s.flags) + ["--max-model-len", str(max_model_len), "--gpu-memory-utilization", str(gpu_memory_utilization)]
    if not g.has_bf16:
        flags += ["--dtype", "half"]
        notes.append(f"{g.name} (sm_{sm}) has no bfloat16: --dtype half (Marlin also prefers fp16 before sm_90)")
    supported, compute, kernel = True, "", ""
    if sm < 75:
        return ServePlan(scheme, g.name, False, model or s.t1_model, flags, "none", "-", s.weight_bits,
                         ["vLLM v0.30.0 kernels start at sm_75"])
    if scheme == "bf16":
        compute, kernel = "16-bit tensor cores", "cuBLAS (unquantized)"
    elif scheme in ("fp8", "fp8-online"):
        if sm >= 89:
            compute = "FP8 x FP8 tensor cores (W8A8); activations quantized per token at run time"
            kernel = "CutlassFP8ScaledMMLinearKernel or FlashInferFP8ScaledMMLinearKernel"
            if sm == 89:
                notes.append("CUTLASS FP8 on sm_89 needs CUDA >= 12.4 (vLLM's image ships CUDA 13)")
        else:
            compute = "weight-only FP8 (W8A16): Marlin dequantizes to 16-bit; memory win only"
            kernel = "MarlinFP8ScaledMMLinearKernel"
            notes.append(f"{g.name} has no FP8 tensor cores: FP8 halves weight bytes but not FLOPs")
        if scheme == "fp8-online":
            notes.append("--quantization fp8_per_tensor quantizes a bf16 checkpoint as it loads "
                         "(`fp8` works in v0.30.0 but raises on main); no calibration, `lm_head` stays 16-bit")
    elif scheme == "w4a16":
        compute = "weight-only INT4: dequantize in registers, 16-bit tensor-core math (bytes win, not FLOPs)"
        kernel = "MacheteLinearKernel" if sm == 90 else "MarlinLinearKernel"
        if sm >= 100:
            notes.append("Machete is Hopper-only (sm_90); Blackwell runs Marlin")
    elif scheme == "w8a8-int8":
        if sm >= 100:
            supported, compute, kernel = False, "refused", "-"
            notes.append("vLLM docs: INT8 W8A8 is not supported on compute capability >= 10.0 (Blackwell); use FP8 or NVFP4")
        else:
            compute = f"INT8 x INT8 tensor cores ({g.tflops.get('int8', 0):g} dense TOPS)"
            kernel = "CutlassInt8ScaledMMLinearKernel"
    elif scheme == "nvfp4":
        if sm >= 100:
            compute = "FP4 x FP4 tensor cores (W4A4), activations quantized per 16 at run time"
            kernel = "FlashInferCutlassNvFp4LinearKernel or CutlassNvFp4LinearKernel"
            notes.append("CUTLASS NVFP4 needs CUDA >= 12.8 and sm_100-sm_12x")
        else:
            compute = "weight-only NVFP4 (Marlin, 16-bit math): memory win only; llm-compressor: '< SM100 ... only weight-only'"
            kernel = "MarlinNvFp4LinearKernel"
    if kv_cache_dtype != "auto":
        from .kv import attention_backend        # local import: kv imports this module's GPU table
        be = attention_backend(g, kv_cache_dtype)
        flags += ["--kv-cache-dtype", kv_cache_dtype]
        if be.error:
            supported = False
            notes.append(be.error)
        else:
            notes.append(f"KV cache {kv_cache_dtype}: attention backend {be.backend}")
    return ServePlan(scheme, g.name, supported, model or s.t1_model, flags, compute, kernel, s.weight_bits, notes)


def matrix(gpus=("T4", "A100-80GB", "L4", "H100-80GB", "B200"), schemes=tuple(SCHEMES)) -> str:
    """A markdown table: every scheme on every GPU."""
    rows = ["| GPU | scheme | runs | what the GEMMs do | vLLM kernel (verify) | flags |", "|---|---|---|---|---|---|"]
    rows += [plan(s, g).table_row() for g in gpus for s in schemes]
    return "\n".join(rows)


# ---------------------------------------------------------------------------------------------
# From a checkpoint's config.json to what vLLM will do with it
# ---------------------------------------------------------------------------------------------
def vllm_scheme(quantization_config: dict) -> dict:
    """Map a compressed-tensors ``quantization_config`` to vLLM's scheme class and its minimum
    compute capability (``compressed_tensors.py: _get_scheme_from_parts``; ``get_min_capability``)."""
    qc = quantization_config
    if qc.get("quant_method") != "compressed-tensors":
        m = qc.get("quant_method", "?")
        mins = {"fp8": 75, "gptq": 75, "awq": 75, "modelopt": 80, "modelopt_fp4": 75, "mxfp4": 80}
        return {"scheme": f"{m} ({qc.get('fmt', '')})", "min_capability": mins.get(m, 75), "w8a8": m == "fp8"}
    grp = next(iter(qc["config_groups"].values()))
    w, a = grp["weights"], grp.get("input_activations")
    fmt = grp.get("format") or qc.get("format")
    if w["type"] == "float" and w["num_bits"] == 4:
        if w.get("group_size") == 16:
            return {"scheme": "CompressedTensorsW4A4Fp4" + ("(use_a16=True)" if a is None else ""),
                    "min_capability": 75, "w8a8": False, "native_from": 100}
        return {"scheme": "CompressedTensorsW4A4Mxfp4", "min_capability": 80, "w8a8": False, "native_from": 100}
    if w["type"] == "int" and a is None and w["strategy"] in ("group", "channel") and fmt == "pack-quantized":
        return {"scheme": "CompressedTensorsWNA16", "min_capability": 75, "w8a8": False}
    if w["type"] == "float" and w["num_bits"] == 8 and a is not None:
        return {"scheme": "CompressedTensorsW8A8Fp8", "min_capability": 75, "w8a8": True, "native_from": 89,
                "fallback": "CompressedTensorsW8A16Fp8 (Marlin, weight-only) below sm_89"}
    if w["type"] == "int" and w["num_bits"] == 8 and a is not None and a["type"] == "int":
        return {"scheme": "CompressedTensorsW8A8Int8", "min_capability": 75, "w8a8": True, "refused_from": 100}
    return {"scheme": "unknown to this table (verify)", "min_capability": 75, "w8a8": False}


def check_checkpoint(config: dict, gpu_name) -> tuple:
    """``(ok, message)`` for serving a checkpoint's ``config.json`` on a GPU, with vLLM's error text."""
    g = gpu(gpu_name)
    qc = config.get("quantization_config")
    if not qc:
        return True, "unquantized: served in --dtype (half on a T4)"
    v = vllm_scheme(qc)
    if g.sm < 70:
        return False, (f"The quantization method {qc.get('quant_method')} is not supported for the current GPU. "
                       f"Minimum capability: 70. Current capability: {g.sm}.")
    if g.sm < v["min_capability"]:
        return False, ("Quantization scheme is not supported for the current GPU. "
                       f"Min capability: {v['min_capability']}. Current capability: {g.sm}.")
    if v.get("refused_from") and g.sm >= v["refused_from"]:
        return False, f"{v['scheme']}: INT8 W8A8 is not supported on compute capability >= 10.0 (vLLM docs)"
    if v.get("native_from") and g.sm < v["native_from"]:
        return True, f"{v['scheme']} loads, but runs weight-only on sm_{g.sm} ({v.get('fallback', 'Marlin')})"
    return True, f"{v['scheme']}: native on sm_{g.sm}"


# ---------------------------------------------------------------------------------------------
# The serving lab's deploy targets, with a quantized model
# ---------------------------------------------------------------------------------------------
def _pairs(flags) -> list:
    return list(zip(flags[0::2], flags[1::2]))          # every flag this module emits takes a value


def cloud_run_tfvars(p: ServePlan, project_id: str = "my-project") -> str:
    """Variables for ``04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/cloud-run/terraform``
    (Cloud Run GPUs are L4 or RTX PRO 6000; this lab plans for the L4)."""
    kv = dict(_pairs(p.flags))
    extra = [x for f, v in _pairs(p.flags) if f not in ("--max-model-len", "--gpu-memory-utilization") for x in (f, v)]
    return "\n".join([f'project_id             = "{project_id}"', f'model                  = "{p.model}"',
                      f"max_model_len          = {kv['--max-model-len']}",
                      f"gpu_memory_utilization = {kv['--gpu-memory-utilization']}",
                      f"extra_args             = {json.dumps(extra)}"]) + "\n"


def gke_container_args(p: ServePlan) -> list:
    """``args`` for the ``vllm`` container in the serving lab's ``deploy/gcp/gke/vllm.yaml``."""
    return [p.model, "--port=8000"] + [f"{f}={v}" for f, v in _pairs(p.flags)]


# ---------------------------------------------------------------------------------------------
# Reading the startup log: did vLLM do what the plan says?
# ---------------------------------------------------------------------------------------------
_LOG = {
    "kernels": r"(?:Using|Selected) (\w+Kernel) for (\w+)",
    "attention_backend": r"Using (\w+) attention backend",
    "kv_cache_dtype": r"Using (\w+) data type to store kv cache",
    "model_loading_gib": r"Model loading took ([\d.]+) GiB",
    "available_kv_gib": r"Available KV cache memory: ([\d.]+) GiB",
    "kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
    "max_concurrency": r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x",
}


def parse_startup_log(text: str) -> dict:
    """The quantization-relevant lines of a ``vllm serve`` log (wording of v0.30.0, verify): the linear
    kernels chosen (``Using MarlinLinearKernel for CompressedTensorsWNA16``), the attention backend, the
    KV dtype, weight memory, KV memory and capacity."""
    import re
    out: dict = {"kernels": sorted(set(re.findall(_LOG["kernels"], text)))}
    for key, pat in _LOG.items():
        if key == "kernels":
            continue
        m = re.search(pat, text)
        if m:
            v = m.group(1).replace(",", "")
            out[key] = v if key in ("attention_backend", "kv_cache_dtype") else (int(v) if key == "kv_cache_tokens" else float(v))
    return out
