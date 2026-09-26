"""stio.py — read and write safetensors files with nothing but the standard library and numpy.

One idea: a ``.safetensors`` file is an 8-byte little-endian header length, a JSON header that
maps each tensor name to ``{"dtype", "shape", "data_offsets"}`` (plus an optional string-to-string
``__metadata__``), then the raw little-endian bytes — which is why a loader can memory-map it and
why a quantized checkpoint is just more tensors with smaller dtypes (``F8_E4M3`` weights, ``I32``
words holding eight INT4 codes, ``BF16`` scales). Numpy has no bfloat16 or float8 types, so those
are carried as their bit patterns (``uint16`` / ``uint8``) and decoded with :mod:`quantlab.numerics`.
The files written here load with the ``safetensors`` package (the tests check it when installed).
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import numerics as N

# safetensors dtype -> numpy storage dtype (bit patterns for the types numpy lacks)
STORAGE = {"F64": np.float64, "F32": np.float32, "F16": np.float16, "BF16": np.uint16,
           "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8,
           "U16": np.uint16, "U32": np.uint32, "BOOL": np.bool_, "F8_E4M3": np.uint8, "F8_E5M2": np.uint8}
NUMPY_TO_ST = {np.dtype(np.float64): "F64", np.dtype(np.float32): "F32", np.dtype(np.float16): "F16",
               np.dtype(np.int64): "I64", np.dtype(np.int32): "I32", np.dtype(np.int16): "I16",
               np.dtype(np.int8): "I8", np.dtype(np.uint8): "U8", np.dtype(np.bool_): "BOOL"}
BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
         "U16": 2, "U32": 4, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


@dataclass
class Tensor:
    """A tensor as stored: its safetensors dtype and a numpy array of the stored bits."""
    dtype: str
    data: np.ndarray

    @property
    def shape(self) -> tuple:
        return tuple(self.data.shape)

    @property
    def nbytes(self) -> int:
        return int(self.data.size * BYTES[self.dtype])

    def numpy(self) -> np.ndarray:
        """Decoded values: float32 for BF16/F8_*, the array itself otherwise."""
        if self.dtype == "BF16":
            return N.bf16_decode(self.data)
        if self.dtype == "F8_E4M3":
            return N.fp8_decode(self.data, "e4m3")
        if self.dtype == "F8_E5M2":
            return N.fp8_decode(self.data, "e5m2")
        return self.data


def bf16(x) -> Tensor:
    return Tensor("BF16", N.bf16_encode(x))


def f8_e4m3(x) -> Tensor:
    return Tensor("F8_E4M3", N.fp8_encode(x, "e4m3"))


def f8_e5m2(x) -> Tensor:
    return Tensor("F8_E5M2", N.fp8_encode(x, "e5m2"))


def as_tensor(x) -> Tensor:
    if isinstance(x, Tensor):
        return x
    a = np.ascontiguousarray(x)
    try:
        return Tensor(NUMPY_TO_ST[a.dtype], a)
    except KeyError:
        raise TypeError(f"no safetensors dtype for numpy {a.dtype}; wrap it with bf16()/f8_e4m3()") from None


def save(tensors: dict, path, metadata: dict | None = None) -> int:
    """Write ``{name: Tensor | ndarray}`` to ``path``; returns the file size in bytes.

    Layout: ``<u64 little-endian N><N bytes of JSON, space-padded to a multiple of 8><data>``,
    tensors laid out back to back in name order, offsets relative to the end of the header."""
    items = sorted((k, as_tensor(v)) for k, v in tensors.items())
    header, offset, blobs = {}, 0, []
    for name, t in items:
        raw = np.ascontiguousarray(t.data).astype(t.data.dtype.newbyteorder("<"), copy=False).tobytes()
        header[name] = {"dtype": t.dtype, "shape": list(t.shape), "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((-len(h)) % 8)
    path = Path(path)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)
    return path.stat().st_size


def read_header(path) -> dict:
    with Path(path).open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def load(path) -> dict:
    """``{name: Tensor}`` from a safetensors file (the ``__metadata__`` entry is skipped)."""
    raw = Path(path).read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + n])
    base = 8 + n
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        dt = np.dtype(STORAGE[info["dtype"]]).newbyteorder("<")
        arr = np.frombuffer(raw, dtype=dt, count=(end - start) // dt.itemsize, offset=base + start)
        out[name] = Tensor(info["dtype"], arr.reshape(info["shape"]).astype(dt.newbyteorder("=")))
    return out


def summary(path) -> list:
    """``[(name, dtype, shape, bytes)]`` from the header alone — what ``safetensors`` metadata tools print."""
    h = read_header(path)
    return [(k, v["dtype"], tuple(v["shape"]), v["data_offsets"][1] - v["data_offsets"][0])
            for k, v in sorted(h.items()) if k != "__metadata__"]
