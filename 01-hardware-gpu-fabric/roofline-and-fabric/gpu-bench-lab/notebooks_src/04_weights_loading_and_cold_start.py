# %% [markdown]
# # 04 · Weights loading and cold start
#
# **Tier:** T0 — write a synthetic safetensors checkpoint and measure *this machine's* storage: cold
# and warm page cache, three read methods, parallel reads. **T1** — stream the file disk → pinned
# host buffer → GPU and compare with the model. **T3** — `deploy/gcp` runs the same suite on a fresh
# cloud VM, whose boot disk is a very different tier from your laptop's SSD. Concepts: primer §6
# "Storage and cold start" ([`../../PRIMER.md`](../../PRIMER.md)).
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
import tempfile

from IPython.display import Markdown, display

from gpubench import get_backend, loading, measure
from gpubench.measure import si
from gpubench.report import measurements_markdown
from gpubench.specs import pcie_gbs

QUICK = True
be = get_backend("auto")
workdir = tempfile.mkdtemp(prefix="gpubench-nb04-")      # point this at the disk you want to measure
path = os.path.join(workdir, "synthetic.safetensors")
print("checkpoint will be written to", path)

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
#   production is cold.
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
disk = max(cold, key=lambda m: m.bytes_per_s()) if cold else None
if disk:
    print(f"best cold read: {si(disk.bytes_per_s(), 'B/s')} ({disk.op}, {disk.params['threads']} thread(s)) "
          f"— this is the number to plan a cold start with")

# %% [markdown]
# Warm reads measure memory copies, not the disk: if warm is several GB/s and cold is not, the page
# cache was doing the work. If cold ≈ warm, the file may live on `tmpfs` (RAM) or the drop was not
# honoured — move `workdir` to a real disk. On a local NVMe, parallel reads usually beat a single
# stream; on a network disk (a cloud boot disk, a FUSE mount of an object store) they are essential.
#
# ## Exercise 4.2 — pipelined vs store-and-forward
#
# Weights pass through several tiers. If each tier finishes before the next starts
# (store-and-forward), times **add**; if chunks stream through all tiers at once (pipelined), the
# **slowest** tier sets the pace. Write `my_load_time(nbytes, tiers, pipelined)` where `tiers` is a
# list of `(name, bytes_per_second)`; return seconds.

# %% exercise
def my_load_time(nbytes, tiers, pipelined=True):
    ### BEGIN SOLUTION
    if pipelined:
        return nbytes / min(bw for _, bw in tiers)
    return sum(nbytes / bw for _, bw in tiers)
    ### END SOLUTION

# %% check
tiers = [("object store", 2e9), ("host RAM staging", 10e9), ("PCIe Gen4", 25e9)]
assert abs(my_load_time(16e9, tiers) - 8.0) < 1e-9
assert abs(my_load_time(16e9, tiers, pipelined=False) - (8.0 + 1.6 + 0.64)) < 1e-9
assert abs(my_load_time(140e9, tiers) - loading.load_time(140e9, tiers)[0]) < 1e-9
print(f"✅ 16 GB: {my_load_time(16e9, tiers):.1f} s streamed vs {my_load_time(16e9, tiers, False):.1f} s hop by hop")

# %% [markdown]
# ## Exercise 4.3 — how many requests in flight?
#
# An object store serves each range request after a first-byte latency, then streams it. To sustain a
# target throughput you need `target × latency ÷ request size` requests outstanding (Little's law).
# Write `my_streams_needed(target_bw, latency_s, request_bytes)` — round **up** to a whole request.

# %% exercise
import math


def my_streams_needed(target_bw, latency_s, request_bytes):
    ### BEGIN SOLUTION
    return math.ceil(target_bw * latency_s / request_bytes)
    ### END SOLUTION

# %% check
assert my_streams_needed(5e9, 0.05, 16e6) == 16                   # 5 GB/s, 50 ms first byte, 16 MB ranges
assert my_streams_needed(1e9, 0.01, 8e6) == 2
assert my_streams_needed(10e9, 0.05, 16e6) == math.ceil(loading.streams_needed(10e9, 0.05, 16e6))
print("✅ fast object-store loaders keep tens of range requests in flight; one stream would crawl")

# %% [markdown]
# ## 3 · To the GPU (T1)
#
# On a GPU host the last hops are host RAM → PCIe → HBM, from a **pinned** buffer so the copy engine
# can DMA it. The lab's loader double-buffers: while chunk *i* is copied to the GPU, chunk *i+1* is
# read into the other pinned buffer — a two-stage pipeline, so the time should approach the slower
# stage, not the sum.

# %%
if disk is not None:
    disk_bw = disk.bytes_per_s()
elif loads:
    disk_bw = max(m.bytes_per_s() for m in loads)
    print("(no cold measurement on this OS: using the best warm read — an optimistic stand-in)")
else:
    disk_bw = None
if be.is_gpu and disk_bw:
    to_dev = []
    for is_cold in (True, False):
        setup = (lambda: loading.drop_page_cache(path)) if is_cold else None
        op = be.make_file_to_device(path, setup=setup)
        to_dev.append(measure(be, op, "load.to_device", {"nbytes": info["bytes"], "cache": "cold" if is_cold else "warm"},
                              repeats=3, min_time=0.0))
    display(Markdown(measurements_markdown(to_dev)))
    pcie = 0.85 * pcie_gbs(4) * 1e9
    print(f"model (pipelined, assuming PCIe Gen4 x16 at 85%): {si(info['bytes'] / min(disk_bw, pcie), 's')}; "
          f"measured cold: {si(to_dev[0].seconds(), 's')}")
elif disk_bw:
    pcie = 0.85 * pcie_gbs(4) * 1e9
    t_pipe = loading.load_time(info["bytes"], [("disk (measured)", disk_bw), ("PCIe Gen4 x16 @85% (model)", pcie)])
    t_sf = loading.load_time(info["bytes"], [("disk (measured)", disk_bw), ("PCIe Gen4 x16 @85% (model)", pcie)], False)
    print(f"No GPU here. Model for this checkpoint on a Gen4 x16 GPU host, with YOUR measured disk: "
          f"{si(t_pipe[0], 's')} pipelined (bottleneck: {t_pipe[1]}) vs {si(t_sf[0], 's')} store-and-forward.\n"
          "On a GPU runtime this cell measures the real disk → pinned → GPU pipeline.")

# %% [markdown]
# ## 4 · The cold-start budget
#
# A new replica pays, in order: provisioning (a VM or node), pulling the container image, loading
# weights, and engine start-up (allocating the KV cache, compiling or capturing CUDA graphs, warm-up).
#
# ## Exercise 4.4 — where does the time go?
#
# Write `cold_start(weight_bytes, weight_tiers, provision_s, image_bytes, image_bw, init_s)` returning
# `(total_seconds, largest_stage_name)` with stages `"provision"`, `"image"`, `"weights"` (pipelined
# through `weight_tiers`, using your `my_load_time`) and `"init"`. The inputs below are illustrative
# round numbers, not measurements.

# %% exercise
def cold_start(weight_bytes, weight_tiers, provision_s, image_bytes, image_bw, init_s):
    ### BEGIN SOLUTION
    stages = {"provision": provision_s, "image": image_bytes / image_bw,
              "weights": my_load_time(weight_bytes, weight_tiers), "init": init_s}
    return sum(stages.values()), max(stages, key=stages.get)
    ### END SOLUTION

# %% check
store = [("object store", 2e9), ("PCIe Gen4", 25e9)]
assert cold_start(16e9, store, 60, 10e9, 0.5e9, 30) == (118.0, "provision")      # an 8B model in bf16
assert cold_start(140e9, store, 60, 10e9, 0.5e9, 30) == (180.0, "weights")       # a 70B model in bf16
total, worst = cold_start(70e9, store, 60, 10e9, 0.5e9, 30)                         # the 70B model in FP8
assert (total, worst) == (145.0, "provision")
print("✅ small models are dominated by fixed costs, big ones by bytes ÷ the slowest tier")

# %%
if disk_bw:
    for label, nbytes in (("8B bf16", 16e9), ("70B bf16", 140e9), ("70B fp8", 70e9)):
        t, worst = cold_start(nbytes, [("this machine's disk (measured)", disk_bw), ("PCIe Gen4", 25e9)], 60, 10e9, 0.5e9, 30)
        print(f"{label:>9} from this machine's disk ({si(disk_bw, 'B/s')}): {t / 60:5.1f} min, dominated by {worst}")
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
