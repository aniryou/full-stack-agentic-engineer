"""quantlab — quantize a checkpoint, serve it, measure what it costs, on the GPU you have (or none).

Modules (each opens with the one idea it teaches):

    compress     recipes -> a compressed-tensors checkpoint (T0 numpy; T1 llm-compressor / GPTQModel)
    serve        which scheme each GPU runs natively, the vLLM kernel, the flags and deploy variables
    bench        TTFT / ITL / throughput per scheme: a roofline engine emulator, a fake server, a real client
    evalharness  lm-evaluation-harness commands and results, plus an offline mini-eval on the tiny model
    kv           --kv-cache-dtype: bytes, blocks, sessions, attention-backend rules, simulated ITL
    fp4          NVFP4 / MXFP4 grids, scales, layouts and a Blackwell throughput model
    report       tables and JSON with every number labelled simulated / measured / sample
    numerics, stio, tinymodel, fakeserver, env   the bits, safetensors I/O, the bundled model, T0 plumbing
"""
import os as _os

# The matrices in this lab are small (<= 256 x 256). A multi-threaded BLAS only adds overhead to
# them — and on a shared or busy machine a great deal (a 256 x 256 inverse went from 7 ms to 1 s
# here). Must happen before numpy is imported; override by setting the variables yourself.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

__version__ = "0.1.0"
