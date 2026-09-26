# %% [markdown]
# # 04 · Weights loading and cold start
#
# **Tier:** T0 — write a synthetic safetensors checkpoint and measure *this machine's* storage: cold
# and warm page cache, three read methods, parallel reads. **T1** — stream the file disk → pinned
# host buffer → GPU and compare with the model. **T3** — `deploy/gcp` runs the same suite on a fresh
# cloud VM, whose boot disk is a very different tier from your laptop's SSD. Concepts: primer §6
# "Storage and cold start" ([`../../PRIMER.md`](../../PRIMER.md)).
#
# **Predicted first in** [roofline-core notebook 04](../../roofline-core/notebooks/04_loading_reliability_and_cost.ipynb)
# (Ex 4.1 streamed loading, Ex 4.2 the cold-start budget) from datasheet tier rates. Here you measure
# the tier you actually have, decide which of your numbers is the one to plan with, and feed it to
# the same model.
#
# ## The one-minute version
#
# A replica is not serving until its weights are in GPU memory, so a cold start is a bandwidth
# problem with a fixed-cost tail: the checkpoint's bytes divided by the **slowest tier** on the
# path (object store → network → disk → page cache → host RAM → PCIe → HBM), plus provisioning,
# image pull and engine start-up. Measure each tier honestly — cold, not warm; with enough requests
# in flight — stream the tiers so the slowest one sets the pace instead of their sum, and you can
# predict, and shorten, how long a new replica takes to come up.

# %%
import os
import shutil

from IPython.display import Markdown, display

from gpubench import get_backend, loading, measure, transfer
from gpubench.accounting import transfer_cost
from gpubench.measure import Measurement, si
from gpubench.report import measurements_markdown
from gpubench.specs import pcie_gbs
from gpubench.timing import Timing

QUICK = True
be = get_backend("auto")
# The disk you measure is the disk this directory is on: point it elsewhere to measure another.
# default_workdir() avoids /tmp when /tmp is tmpfs (RAM), as it is on many distributions.
workdir = loading.default_workdir(prefix="gpubench-nb04-")
path = os.path.join(workdir, "synthetic.safetensors")
print(f"checkpoint will be written to {path} (filesystem: {loading.filesystem_type(workdir) or 'unknown'})")

# %% [markdown]
# ## 1 · The safetensors format
#
# ```
#  8 bytes          N bytes of JSON (space-padded)                    the tensors, back to back
# ┌────────┬──────────────────────────────────────────────┬──────────────────────────────────────┐
# │ N (u64 │ {"model.embed_tokens.weight": {"dtype":"BF16",│ embed_tokens │ layers.0.q_proj │ ...  │
# │ little │   "shape":[32000,1024],"data_offsets":[0,N0]},│              │                 │      │
# │ endian)│  ..., "__metadata__": {...}}                  │ ^offset 0     ^N0               ^... │
# └────────┴──────────────────────────────────────────────┴──────────────────────────────────────┘
# ```
#
# No pickle, so loading runs no code; and every tensor's position is known from the header before
# a byte of data is read, so a loader can `mmap` the file, read tensors in parallel, or stream them
# straight to a GPU. The lab writes it from scratch (`gpubench.loading.write_safetensors`, ~40
# lines); the `safetensors` library reads what it writes. Random bytes fill the tensors so a
# compressing or deduplicating filesystem cannot flatter the measurement.

# %%
info = loading.synthetic_checkpoint(path, (256 << 20) if QUICK else (2 << 30))
header, data_start = loading.read_header(path)
print(f"{info['tensors']} tensors, {info['layers']} layers of hidden {info['hidden']}, {si(info['bytes'], 'B')} "
      f"on disk; the data starts at byte {data_start}")
for name in list(header)[:3]:
    print(f"   {name}: {header[name]}")

# %% [markdown]
# ## Exercise 4.1 — read the header yourself
#
# Write `my_read_header(path)` → `(header_dict, data_start)` (unpack the first 8 bytes as a
# little-endian unsigned 64-bit integer, then parse that many bytes of JSON), and
# `my_nbytes(dtype, shape)` for the dtypes `F64 F32 F16 BF16 F8_E4M3 I64 I32 U8`.

# %% exercise
import json
import struct


def my_read_header(path):
    ### BEGIN SOLUTION
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)), 8 + n
    ### END SOLUTION


def my_nbytes(dtype, shape):
    ### BEGIN SOLUTION
    size = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "I64": 8, "I32": 4, "U8": 1}[dtype]
    count = 1
    for d in shape:
        count *= d
    return count * size
    ### END SOLUTION

# %% check
h, start = my_read_header(path)
assert (h, start) == (header, data_start) and start % 8 == 0
tensors = {k: v for k, v in h.items() if k != "__metadata__"}
assert all(v["data_offsets"][1] - v["data_offsets"][0] == my_nbytes(v["dtype"], v["shape"]) for v in tensors.values())
assert sum(my_nbytes(v["dtype"], v["shape"]) for v in tensors.values()) == os.path.getsize(path) - start
assert my_nbytes("BF16", [4096, 4096]) == 33_554_432 and my_nbytes("F8_E4M3", []) == 1
print(f"✅ header parsed: {len(tensors)} tensors whose sizes add up to every data byte in the file")

# %% [markdown]
# ## 2 · Measuring load throughput honestly
#
# Three pitfalls make load benchmarks lie, and the lab avoids each:
#
# * **The page cache.** Read a file twice and the second read comes from RAM. *Cold* runs evict the
#   file first (`posix_fadvise(DONTNEED)`, Linux, no root needed); *warm* runs do not. A new node in
#   production is cold. A file on `tmpfs` cannot be evicted — it *is* RAM — so the lab refuses to
#   call any read of it cold and lists the cold rows as skipped instead.
# * **First-touch page faults.** Reading into a freshly allocated buffer pays a page fault per 4 KB
#   of destination — billed to "the disk". The lab allocates and touches the buffer once, up front
#   (on a GPU host, that buffer is the pinned staging area).
# * **One request at a time.** SSDs and network disks reach their rated throughput only with many
#   requests in flight — Little's law again. `pread` × threads keeps several outstanding.
#
# Methods: `read` (one thread, sequential `readinto`), `pread` (threads reading disjoint ranges),
# `mmap` (map the file, then copy it out — the mapping is free, the page faults are not).

# %%
skipped = []
loads = loading.bench_load(path, threads=(4,) if QUICK else (2, 4, 8), repeats=3, skipped=skipped)
display(Markdown(measurements_markdown(loads)))
for s in skipped:
    print("skipped:", s)
cold = [m for m in loads if m.params["cache"] == "cold"]
warm = [m for m in loads if m.params["cache"] == "warm"]

# %% [markdown]
# ## Exercise 4.2 — is your "cold" number a disk number?
#
# Before planning with a measurement, decide what it measured. Write `cold_verdict(cold_bps,
# warm_bps, filesystem)` returning:
#
# * `"ram"` if the file lives on `tmpfs` or `ramfs` — every read is a memory copy;
# * `"suspect"` if the cold read ran at 70% or more of the warm read — either the eviction was not
#   honoured (some container and network filesystems ignore it) or the device is as fast as memory
#   copies (rare); check with `O_DIRECT` or a file larger than RAM before trusting it;
# * `"disk"` otherwise — the number to plan a cold start with.

# %% exercise
def cold_verdict(cold_bps, warm_bps, filesystem):
    ### BEGIN SOLUTION
    if filesystem in ("tmpfs", "ramfs"):
        return "ram"
    return "suspect" if cold_bps >= 0.7 * warm_bps else "disk"
    ### END SOLUTION

# %% check
assert cold_verdict(1.5e9, 9e9, "ext4") == "disk"                  # an NVMe behind a fast page cache
assert cold_verdict(8.5e9, 9e9, "overlay") == "suspect"            # "cold" as fast as warm: not evicted?
assert cold_verdict(9e9, 9e9, "tmpfs") == "ram"
fs = loading.filesystem_type(path)
by = {(m.op, m.params["cache"]): m for m in loads if m.params["threads"] in (1,)}
if ("load.read", "cold") in by:
    c, w = by[("load.read", "cold")].bytes_per_s(), by[("load.read", "warm")].bytes_per_s()
    print(f"this machine ({fs}): cold read {si(c, 'B/s')}, warm {si(w, 'B/s')} → {cold_verdict(c, w, fs)}")
else:
    print(f"no cold rows on this machine ({fs}):", "; ".join(s["reason"] for s in skipped))
print("✅ a cold read is only a disk number if the bytes really came from the disk")

# %% [markdown]
# Warm reads measure memory copies, not the disk: if warm is several GB/s and cold is not, the page
# cache was doing the work. On a local NVMe, parallel reads usually beat a single stream; on a
# network disk (a cloud boot disk, a FUSE mount of an object store) they are essential — Little's
# law again: an object store serving each range request after a first-byte latency needs
# `target × latency ÷ request size` requests outstanding (`loading.streams_needed`; roofline-core
# Ex 4.2 sizes it). For 5 GB/s at 50 ms per 16 MB range that is 5e9 × 0.05 ÷ 16e6 ≈ 16 requests in
# flight, not one.
#
# ## Exercise 4.3 — which measured rate belongs in the model?
#
# The model is the one from primer §6 (`loading.load_time`): pipelined tiers run at the slowest
# tier's rate. The judgement is which of your measurements *is* that tier. The lab's disk → GPU
# loader (`make_file_to_device`, section 3) reads the file with **one** sequential stream, cold on a
# new node — so its disk tier is the cold, one-thread `load.read` row, not the fastest row in the
# table (a 4-thread `pread` or, worse, a warm read would flatter the prediction). Write
# `loader_disk_rate(measurements)`: the best-sample bytes/s of that row, or `None` if there is none.

# %% exercise
def loader_disk_rate(measurements):
    ### BEGIN SOLUTION
    rows = [m for m in measurements
            if m.op == "load.read" and m.params["cache"] == "cold" and m.params["threads"] == 1]
    return max((m.bytes_per_s() for m in rows), default=None)
    ### END SOLUTION

# %% check
def _row(method, cache, threads, gbs):
    return Measurement(f"load.{method}", {"nbytes": 1e9, "cache": cache, "threads": threads},
                       transfer_cost(1e9), Timing((1 / gbs,)), "host", "cpu")

fake = [_row("read", "cold", 1, 1.2), _row("pread", "cold", 4, 3.1), _row("read", "warm", 1, 9.0)]
assert abs(loader_disk_rate(fake) - 1.2e9) < 1
assert loader_disk_rate([_row("read", "warm", 1, 9.0)]) is None
disk_bw = loader_disk_rate(loads)
print(f"✅ the loader's disk tier on this machine: {si(disk_bw, 'B/s') if disk_bw else 'not measurable here (no cold rows)'}")

# %% [markdown]
# ## 3 · To the GPU (T1)
#
# On a GPU host the last hops are host RAM → PCIe → HBM, from a **pinned** buffer so the copy engine
# can DMA it. The lab's loader double-buffers: while chunk *i* is copied to the GPU, chunk *i+1* is
# read into the other pinned buffer — a two-stage pipeline, so the time should approach the slower
# stage, not the sum.

# %%
link = transfer.host_link(be.describe().get("name") if be.is_gpu else None)
pcie = 0.85 * pcie_gbs(link["gen"], link["width"]) * 1e9
pcie_label = f"PCIe Gen{link['gen']} x{link['width']} @85% ({link['source']})"
if disk_bw is None:
    print("No cold one-thread read was measurable here (see the skipped reasons above), so there is no disk tier "
          "to model with: set workdir to a directory on a real disk and re-run.")
else:
    tiers = [("disk: cold one-thread read (measured)", disk_bw), (pcie_label, pcie)]
    t_pipe, slowest = loading.load_time(info["bytes"], tiers)
    t_sf, _ = loading.load_time(info["bytes"], tiers, pipelined=False)
    for name, rate_ in tiers:
        print(f"   tier: {name:<50} {si(rate_, 'B/s')}")
    print(f"model for this {si(info['bytes'], 'B')} checkpoint: {si(t_pipe, 's')} pipelined (bottleneck: {slowest}) "
          f"vs {si(t_sf, 's')} store-and-forward")
    if be.is_gpu:
        cold_ok, why = loading.cold_read_possible(path)
        to_dev = []
        for is_cold in ((True, False) if cold_ok else (False,)):
            setup = (lambda: loading.drop_page_cache(path)) if is_cold else None
            op = be.make_file_to_device(path, setup=setup)
            to_dev.append(measure(be, op, "load.to_device", {"nbytes": info["bytes"],
                                                             "cache": "cold" if is_cold else "warm"},
                                  repeats=3, min_time=0.0))
        display(Markdown(measurements_markdown(to_dev)))
        print(f"measured {to_dev[0].params['cache']}: {si(to_dev[0].seconds(), 's')} against the pipelined model's "
              f"{si(t_pipe, 's')} and the store-and-forward {si(t_sf, 's')}")
    else:
        print("No GPU here: the PCIe tier above is a model of a GPU host's link, not a measurement. On a GPU runtime "
              "(Colab: Runtime → Change runtime type → T4 GPU) this cell also measures the real disk → pinned → GPU "
              "pipeline, with the PCIe generation and width nvidia-smi reports.")

# %% [markdown]
# If the measured pipeline lands near the pipelined model, double-buffering works and the disk is the
# bottleneck; near the store-and-forward sum, the two stages are not overlapping (the reader waits for
# the copy engine, or the copy waits for the reader). A loader that reads with several threads (a
# `pread`-style reader, or a streaming loader for object stores) would move the disk tier to the
# faster rows of section 2 — which is the next exercise.
#
# ## Exercise 4.4 — pick the loader for this disk
#
# From your cold measurements, choose how a loader on this machine should read: write
# `pick_loader(measurements)` → `(method, threads)`. Take the fastest **cold** row, but keep the
# simple one-stream `("read", 1)` unless the winner beats it by more than 20% — extra threads and
# memory maps are complexity you should only pay for when the disk rewards them. Return `None` if
# there are no cold rows.

# %% exercise
def pick_loader(measurements):
    ### BEGIN SOLUTION
    cold_rows = [m for m in measurements if m.params["cache"] == "cold"]
    if not cold_rows:
        return None
    best = max(cold_rows, key=lambda m: m.bytes_per_s())
    base = max((m.bytes_per_s() for m in cold_rows if m.op == "load.read"), default=0.0)
    if best.bytes_per_s() > 1.2 * base:
        return best.op.split(".", 1)[1], best.params["threads"]
    return "read", 1
    ### END SOLUTION

# %% check
nvme = [_row("read", "cold", 1, 1.5), _row("pread", "cold", 4, 4.2), _row("mmap", "cold", 1, 1.1), _row("read", "warm", 1, 9)]
assert pick_loader(nvme) == ("pread", 4)                      # parallel requests pay on an SSD
flat = [_row("read", "cold", 1, 0.25), _row("pread", "cold", 4, 0.27), _row("mmap", "cold", 1, 0.2)]
assert pick_loader(flat) == ("read", 1)                       # one stream already saturates this disk
assert pick_loader([_row("read", "warm", 1, 9)]) is None
print(f"✅ this machine: {pick_loader(loads) or 'no cold rows — measure on a real disk'}")

# %% [markdown]
# ## 4 · The cold-start budget
#
# A new replica pays, in order: provisioning (a VM or node), pulling the container image, loading
# weights, and engine start-up (allocating the KV cache, compiling or capturing CUDA graphs, warm-up)
# — primer §6.2; roofline-core Ex 4.2 budgets it from datasheet rates. Here only the weights stage
# uses a measurement (your loader's disk tier); the other stages are **assumed round numbers**, and
# the table says so.

# %%
ASSUMED = {"provision_s": 60, "image_bytes": 10e9, "image_bps": 0.5e9, "init_s": 30}   # illustrative, not measured
if disk_bw:
    rate = disk_bw
    choice = pick_loader(loads)
    if choice and choice != ("read", 1):
        rate = max(m.bytes_per_s() for m in loads if m.params["cache"] == "cold"
                   and m.op == f"load.{choice[0]}" and m.params["threads"] == choice[1])
    print(f"weights: this machine's cold {choice[0] if choice else 'read'} rate {si(rate, 'B/s')} (measured), "
          f"then {pcie_label} — pipelined")
    print(f"assumed: provision {ASSUMED['provision_s']} s, image {si(ASSUMED['image_bytes'], 'B')} at "
          f"{si(ASSUMED['image_bps'], 'B/s')}, engine init {ASSUMED['init_s']} s\n")
    for label, nbytes in (("8B bf16", 16e9), ("70B bf16", 140e9), ("70B fp8", 70e9)):
        weights_s, _ = loading.load_time(nbytes, [("disk", rate), ("pcie", pcie)])
        cs = loading.cold_start({"provision": ASSUMED["provision_s"], "image": ASSUMED["image_bytes"] / ASSUMED["image_bps"],
                                 "weights": weights_s, "init": ASSUMED["init_s"]})
        print(f"{label:>9}: {cs['total_s'] / 60:5.1f} min, largest stage {cs['largest']} "
              f"({cs['shares'][cs['largest']]:.0%}) — weights measured-disk, other stages assumed")
shutil.rmtree(workdir, ignore_errors=True)       # the synthetic checkpoint is not needed any more

# %% [markdown]
# ## In a design review
#
# **The two-minute version.** "Cold start is bytes over the slowest tier plus fixed costs. For a
# 70B model in bf16 that is 140 GB: at a cloud boot disk's few hundred MB/s it is many minutes, at a
# local NVMe or a well-parallelised object-store stream a minute or less. I measure each tier cold
# (a new node has an empty page cache), with enough requests in flight (Little's law), and stream
# through the tiers so the slowest sets the pace. safetensors makes that possible — offsets up
# front, no unpickling, zero-copy views — and FP8 weights halve the dominant term. Anything left
# over is provisioning, image pull and engine init, which is where warm pools and image streaming
# come in (layers 03 and 05)."
#
# **Drills**
#
# 1. *New replicas of our 70B model take nine minutes. Where do you look first?* — Break it into
#    stages. 140 GB at ~260 MB/s is nine minutes on its own, so the weights path is the suspect:
#    measure the disk cold; move weights to a faster tier (local NVMe, a high-throughput volume,
#    parallel range reads from the object store); stream instead of staging; consider FP8.
# 2. *Why does a safetensors checkpoint load faster than a pickled one?* — No unpickling or object
#    construction; the header gives every tensor's offset, so the loader can mmap, read in parallel
#    and copy straight into pinned buffers — and loading executes no code, which is also a security
#    property.
# 3. *Our loader benchmark shows 12 GB/s from an NVMe rated at 7 GB/s.* — It is reading the page
#    cache. Evict the file (or use O_DIRECT) and measure again; production cold starts are cold.
