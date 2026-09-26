"""Weights loading: the safetensors format from scratch, disk throughput measured honestly, and
the cold-start budget.

**The format.** A ``.safetensors`` file is an 8-byte little-endian header length ``N``, then ``N``
bytes of JSON (``{name: {"dtype", "shape", "data_offsets": [begin, end]}, "__metadata__": {...}}``),
then the raw tensor bytes, back to back. No pickle, no code execution, and every tensor's
position is known before a byte of data is read — so a loader can ``mmap`` the file, read
tensors in parallel, or stream them to a GPU. ``write_safetensors``/``read_header`` implement it
in ~40 lines, and the tests check them against the ``safetensors`` library (when installed)
in both directions.

**The measurement.** "How fast can I load weights?" has three honest answers, and the lab
measures each: *cold* (the file is not in the OS page cache — the disk's speed; the lab evicts
the file with ``posix_fadvise(DONTNEED)``, no root needed, Linux only — and refuses to call a
read cold when the file sits on ``tmpfs``/``ramfs``, which *is* RAM and cannot be evicted; ``/tmp``
is tmpfs on many distributions, so ``default_workdir`` avoids it), *warm* (the page cache
has it — you are measuring memory copies), and *parallel* (several reads in flight — what SSDs
and network disks need to reach their rated throughput: Little's law again). The destination
buffer is allocated and touched once, up front: otherwise the first-touch page faults on a fresh
buffer get billed to the disk (the host-side cousin of pinned memory).

**The model.** A cold start is a sum of stages (provision, image pull, weights, engine init);
the weights stage is ``bytes / slowest tier`` when tiers are pipelined (streamed chunk by chunk)
and ``Σ bytes / tier`` when each hop finishes before the next starts (primer §6).
"""
from __future__ import annotations

import json
import mmap
import os
import re
import struct
import tempfile

import numpy as np

from .accounting import transfer_cost
from .measure import Op, measure

ST_DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1, "I16": 2, "U16": 2, "F16": 2,
                  "BF16": 2, "I32": 4, "U32": 4, "F32": 4, "F64": 8, "I64": 8, "U64": 8}
# numpy has no bfloat16/fp8: view those as raw integers of the same width
NUMPY_VIEW = {"BOOL": "?", "U8": "u1", "I8": "i1", "F8_E5M2": "u1", "F8_E4M3": "u1", "I16": "<i2", "U16": "<u2",
              "F16": "<f2", "BF16": "<u2", "I32": "<i4", "U32": "<u4", "F32": "<f4", "F64": "<f8", "I64": "<i8",
              "U64": "<u8"}
MAX_HEADER = 100_000_000


def tensor_nbytes(dtype: str, shape) -> int:
    """Bytes of one tensor: product of the shape (1 for a scalar) × bytes per element."""
    n = 1
    for d in shape:
        n *= int(d)
    return n * ST_DTYPE_BYTES[dtype]


def _as_bytes(data) -> memoryview:
    if callable(data):
        data = data()
    if isinstance(data, np.ndarray):
        data = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
    return memoryview(data).cast("B")


def write_safetensors(path, tensors: dict, metadata: dict | None = None, align: int = 8) -> int:
    """Write ``{name: (dtype, shape, data)}`` in insertion order; ``data`` is bytes-like, an array,
    or a zero-argument callable returning one (so huge files are generated tensor by tensor).
    The header is padded with spaces so the data starts ``align``-byte aligned. Returns bytes written."""
    header, offset = {}, 0
    for name, (dtype, shape, _) in tensors.items():
        nb = tensor_nbytes(dtype, shape)
        header[name] = {"dtype": dtype, "shape": [int(d) for d in shape], "data_offsets": [offset, offset + nb]}
        offset += nb
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    hbytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hbytes += b" " * ((-(8 + len(hbytes))) % align)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hbytes)))
        f.write(hbytes)
        for name, (dtype, shape, data) in tensors.items():
            buf = _as_bytes(data)
            if len(buf) != tensor_nbytes(dtype, shape):
                raise ValueError(f"{name}: {len(buf)} bytes given, {tensor_nbytes(dtype, shape)} expected")
            f.write(buf)
        f.flush()
        os.fsync(f.fileno())          # on disk, and clean in the page cache (so it can be evicted)
    return 8 + len(hbytes) + offset


def read_header(path) -> tuple:
    """``(header_dict, data_start)``: parse the 8-byte length and the JSON header."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        if n > MAX_HEADER:
            raise ValueError(f"header of {n} bytes is implausible (limit {MAX_HEADER})")
        header = json.loads(f.read(n))
    return header, 8 + n


def validate(header: dict, file_size: int, data_start: int) -> list:
    """Problems with a header (empty list = valid): unknown dtypes, sizes that disagree with
    shapes, and data offsets that leave holes or overlap or run past the file."""
    problems, spans = [], []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype, (b, e) = info["dtype"], info["data_offsets"]
        if dtype not in ST_DTYPE_BYTES:
            problems.append(f"{name}: unknown dtype {dtype}")
            continue
        if e - b != tensor_nbytes(dtype, info["shape"]):
            problems.append(f"{name}: offsets span {e - b} bytes, shape needs {tensor_nbytes(dtype, info['shape'])}")
        spans.append((b, e, name))
    pos = 0
    for b, e, name in sorted(spans):
        if b != pos:
            problems.append(f"{name}: starts at {b}, expected {pos} (hole or overlap)")
        pos = max(pos, e)
    if pos != file_size - data_start:
        problems.append(f"data ends at {pos}, file holds {file_size - data_start} data bytes")
    return problems


def synthetic_checkpoint(path, target_bytes: int, hidden: int = 1024, dtype: str = "BF16", vocab: int = 32000,
                         seed: int = 0) -> dict:
    """A transformer-shaped checkpoint of about ``target_bytes`` filled with random bytes (random
    so a compressing or deduplicating filesystem cannot flatter the read). Returns a summary."""
    rng = np.random.default_rng(seed)
    inter = int(hidden * 2.75) // 64 * 64
    tensors = {"model.embed_tokens.weight": (dtype, (vocab, hidden))}
    per_layer = [("self_attn.q_proj", (hidden, hidden)), ("self_attn.k_proj", (hidden, hidden)),
                 ("self_attn.v_proj", (hidden, hidden)), ("self_attn.o_proj", (hidden, hidden)),
                 ("mlp.gate_proj", (inter, hidden)), ("mlp.up_proj", (inter, hidden)),
                 ("mlp.down_proj", (hidden, inter)), ("input_layernorm", (hidden,)),
                 ("post_attention_layernorm", (hidden,))]
    total, layer = tensor_nbytes(dtype, (vocab, hidden)), 0
    while total < target_bytes:
        for suffix, shape in per_layer:
            tensors[f"model.layers.{layer}.{suffix}.weight"] = (dtype, shape)
            total += tensor_nbytes(dtype, shape)
        layer += 1
    spec = {name: (dt, shape, (lambda nb=tensor_nbytes(dt, shape): rng.bytes(nb))) for name, (dt, shape) in tensors.items()}
    written = write_safetensors(path, spec, metadata={"format": "pt", "generator": "gpubench synthetic"})
    return {"path": str(path), "bytes": written, "tensors": len(tensors), "layers": layer, "hidden": hidden, "dtype": dtype}


# -- where the file lives ------------------------------------------------------------------------------------
RAM_FILESYSTEMS = {"tmpfs", "ramfs"}


def filesystem_type(path, mountinfo: str | None = None) -> str | None:
    """The filesystem type holding ``path`` (``"ext4"``, ``"tmpfs"``, ...) from the longest matching
    mount point in ``/proc/self/mountinfo`` (or the ``mountinfo`` text given); None where that file
    does not exist (not Linux)."""
    if mountinfo is None:
        try:
            mountinfo = open("/proc/self/mountinfo").read()
        except OSError:
            return None
    target = os.path.realpath(path)
    best, fstype = -1, None
    for line in mountinfo.splitlines():
        left, sep, right = line.partition(" - ")
        fields = left.split()
        if not sep or len(fields) < 5 or not right.split():
            continue
        mnt = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), fields[4])   # \040 is a space
        inside = target == mnt or mnt == "/" or target.startswith(mnt.rstrip("/") + "/")
        if inside and len(mnt) >= best:           # deepest mount point; a later mount over it wins
            best, fstype = len(mnt), right.split()[0]
    return fstype


def cold_read_possible(path) -> tuple:
    """``(True, "")`` if a read of ``path`` can be made cold here, else ``(False, why)``."""
    if not hasattr(os, "posix_fadvise"):
        return False, "posix_fadvise is unavailable (not Linux): the page cache cannot be dropped"
    fs = filesystem_type(path)
    if fs in RAM_FILESYSTEMS:
        return False, (f"the file is on {fs}, i.e. in RAM: nothing can be evicted, so a 'cold' read would be a "
                       "memory copy — point the workdir at a real disk")
    return True, ""


def default_workdir(prefix: str = "gpubench-") -> str:
    """A fresh temporary directory on a real disk: the system temp dir unless it is RAM-backed
    (``/tmp`` is tmpfs on Fedora, Arch, Debian 13, ...), else the current directory, else home."""
    for base in (tempfile.gettempdir(), os.getcwd(), os.path.expanduser("~")):
        if filesystem_type(base) not in RAM_FILESYSTEMS and os.access(base, os.W_OK):
            return tempfile.mkdtemp(prefix=prefix, dir=base)
    return tempfile.mkdtemp(prefix=prefix)


# -- reading, three ways ----------------------------------------------------------------------------------
def drop_page_cache(path) -> bool:
    """Evict ``path`` from the OS page cache (Linux ``posix_fadvise``; no root needed). Returns
    False where that cannot make the next read cold — no ``posix_fadvise``, or a file on
    tmpfs/ramfs (RAM itself) — and then the lab does not call the read cold."""
    if not cold_read_possible(path)[0]:
        return False
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def read_sequential(path, dst: np.ndarray, chunk_bytes: int = 16 << 20) -> int:
    """One thread, ``readinto`` chunk after chunk (unbuffered: no extra copy through Python)."""
    mv = memoryview(dst).cast("B")
    size = os.path.getsize(path)
    with open(path, "rb", buffering=0) as f:
        off = 0
        while off < size:
            r = f.readinto(mv[off:min(size, off + chunk_bytes)])
            if not r:
                raise IOError(f"short read at {off} of {path}")
            off += r
    return off


def read_parallel(path, dst: np.ndarray, threads: int = 4, chunk_bytes: int = 16 << 20) -> int:
    """``threads`` workers read disjoint ranges concurrently (``os.preadv``): more requests in flight."""
    from .backends.numpy_backend import _pool

    mv = memoryview(dst).cast("B")
    size = os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY)

    def job(off: int) -> int:
        end, got = min(size, off + chunk_bytes), 0
        while off + got < end:
            if hasattr(os, "preadv"):
                r = os.preadv(fd, [mv[off + got:end]], off + got)
            else:                                       # platforms without preadv
                data = os.pread(fd, end - off - got, off + got)
                mv[off + got:off + got + len(data)] = data
                r = len(data)
            if not r:
                raise IOError(f"short read at {off + got} of {path}")
            got += r
        return got

    try:
        return sum(_pool(threads).map(job, range(0, size, chunk_bytes)))
    finally:
        os.close(fd)


def read_mmap(path, dst: np.ndarray) -> int:
    """Map the file and copy it out. Mapping alone is nearly free — the cost arrives as page
    faults when the bytes are first touched, which the copy does."""
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            src = np.frombuffer(mm, dtype=np.uint8)
            np.copyto(dst[: src.size], src)
            n = src.size
            del src                                     # release the buffer before closing the map
        finally:
            mm.close()
    return n


METHODS = {"read": read_sequential, "pread": read_parallel, "mmap": read_mmap}


def make_load_op(path, method: str = "read", threads: int = 1, chunk_bytes: int = 16 << 20, cold: bool = False,
                 dst: np.ndarray | None = None) -> Op:
    """An Op that reads the whole file into a pre-touched host buffer by ``method``."""
    size = os.path.getsize(path)
    if dst is None:
        dst = np.empty(size, dtype=np.uint8)
    dst.fill(0)                                        # first-touch page faults happen here, not in the timing
    if method == "read":
        fn = lambda: read_sequential(path, dst, chunk_bytes)  # noqa: E731
    elif method == "pread":
        fn = lambda: read_parallel(path, dst, threads, chunk_bytes)  # noqa: E731
    elif method == "mmap":
        fn = lambda: read_mmap(path, dst)  # noqa: E731
    else:
        raise ValueError(f"method must be one of {list(METHODS)}")
    setup = (lambda: drop_page_cache(path)) if cold else None
    cold_ok, why = cold_read_possible(path) if cold else (True, "")
    if not cold:
        note = "warm (page cache)"
    elif cold_ok:
        note = "cold (page cache dropped before each sample)"
    else:
        note = f"NOT cold — {why}"

    def verify() -> bool:
        with open(path, "rb") as f:
            head = f.read(1 << 16)
            f.seek(max(0, size - (1 << 16)))
            tail = f.read()
        return bytes(dst[: len(head)]) == head and bytes(dst[size - len(tail): size]) == tail

    return Op(fn, transfer_cost(size), setup=setup, max_inner=1 if cold else None, verify=verify, note=note,
              extras={"cache": "cold" if cold else "warm", "cold_valid": cold_ok,
                      "threads": threads if method == "pread" else 1, "chunk_bytes": chunk_bytes,
                      "filesystem": filesystem_type(path)})


def bench_load(path, methods=("read", "pread", "mmap"), caches=("cold", "warm"), threads=(4,),
               chunk_bytes: int = 16 << 20, repeats: int = 3, skipped: list | None = None) -> list:
    """Measure each method × cache state (× thread count for ``pread``) on ``path``."""
    size = os.path.getsize(path)
    dst = np.empty(size, dtype=np.uint8)
    cold_ok, why = cold_read_possible(path)
    out = []
    for cache in caches:
        if cache == "cold" and not cold_ok:
            if skipped is not None:
                skipped.append({"what": "cold reads", "reason": why})
            continue
        for method in methods:
            for t in (threads if method == "pread" else (1,)):
                op = make_load_op(path, method, t, chunk_bytes, cache == "cold", dst)
                out.append(measure(None, op, f"load.{method}",
                                   {"nbytes": size, "cache": cache, "threads": t}, repeats=repeats, min_time=0.0))
    return out


def load_tensors(path, dst: np.ndarray | None = None) -> dict:
    """Read a safetensors file and return ``{name: numpy view}`` — zero-copy views into one buffer
    (BF16/FP8 appear as raw uint16/uint8: numpy has no such dtypes)."""
    header, start = read_header(path)
    size = os.path.getsize(path)
    buf = dst if dst is not None else np.empty(size, dtype=np.uint8)
    read_sequential(path, buf)
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        b, e = info["data_offsets"]
        out[name] = buf[start + b:start + e].view(NUMPY_VIEW[info["dtype"]]).reshape(info["shape"])
    return out


# -- the cold-start model ------------------------------------------------------------------------------------
def load_time(nbytes: float, tiers, pipelined: bool = True) -> tuple:
    """Seconds to move ``nbytes`` through ``tiers`` = ``[(name, bytes_per_s), ...]``.

    Pipelined (chunks stream through every tier at once): ``nbytes / slowest``.
    Store-and-forward (each tier finishes before the next starts): ``Σ nbytes / tier``.
    Returns ``(seconds, name of the slowest tier)``.
    """
    slowest = min(tiers, key=lambda t: t[1])
    if pipelined:
        return nbytes / slowest[1], slowest[0]
    return sum(nbytes / bw for _, bw in tiers), slowest[0]


def streams_needed(target_bw: float, latency_s: float, request_bytes: float) -> float:
    """Little's law for object storage: requests in flight = target throughput × latency ÷ request size."""
    return target_bw * latency_s / request_bytes


def cold_start(stages: dict) -> dict:
    """Sum a cold start's stages (seconds) and name the one to attack first."""
    total = sum(stages.values())
    largest = max(stages, key=stages.get)
    return {"total_s": total, "largest": largest, "shares": {k: v / total for k, v in stages.items()}}
