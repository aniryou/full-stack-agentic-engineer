# %% [markdown]
# # 06 · Deploy on Cloud Run GPU: cold starts, concurrency and cost, planned before you apply
#
# **Tier:** T3 walkthrough (GCP), and you can do all of the plan at T0. This notebook reads the Terraform and
# the `gcloud` script and does a dry run of them. The engine is the fake vLLM with an **L4 profile** (the
# results are **simulated**). Each price or bandwidth is an explicit *assumption* that you replace. With
# `gcloud`, a project and `SERVELAB_DEPLOY=1`, the last cell deploys for real. Then set `SERVELAB_URL` and
# run notebook 02 again.
#
# ## The one-minute version
#
# A Cloud Run GPU service is a pool of identical instances. Each instance is one engine on one L4 (24 GB).
# Three settings decide the latency and the cost:
#
# * **`concurrency`**: the number of requests that Cloud Run sends to one instance. Set it equal to the
#   batch that the engine can serve *within the SLO*. With a larger value, requests wait in a queue inside
#   the engine, and TPOT and TTFT become worse. With a smaller value, Cloud Run scales out earlier than
#   necessary, and the cost increases.
# * **`min_instances`**: 0 means scale to zero. There is no cost while the service is idle. But the first
#   request after an idle period gets a **cold start** (image pull, weights and engine init). 1 means that
#   the instance is always warm and always billed.
# * **`max_instances`**: a limit on the GPUs, on the bill, and on the load that you can absorb.
#
# Concepts: PRIMER §12 "Engines and where to run them" ([`PRIMER.md`](../../PRIMER.md)). The deploy
# assets are in [`deploy/gcp/cloud-run/`](../deploy/gcp/cloud-run/) (Terraform and `gcloud`). For the
# prices, see `COMPUTE.md` at the repo root. Admission control, rate limits and cost per conversation in
# front of a fleet like this one are the subjects of layer 06:
# [`agentic-scaling-lab`](../../../../06-gateway/scaling-admission-cost/agentic-scaling-lab/).

# %%
import math, os, re, shutil, subprocess
from pathlib import Path
import servelab
from servelab import env, sizing
from servelab.bench import SLO, Lengths, random_requests, run_closed_loop
from servelab.bench.summary import Stat, Summary
from servelab.fake_engine import EngineConfig
from servelab.fakeserver import FakeServer

LAB = Path(servelab.__file__).resolve().parents[1]
CR = LAB / "deploy" / "gcp" / "cloud-run"
print(env.describe())
tf = (CR / "terraform" / "cloud_run.tf").read_text()
keys = ("accelerator", "nvidia.com/gpu", "max_instance_request_concurrency", "min_instance_count",
        "max_instance_count", "gpu_zonal_redundancy_disabled", "path", "failure_threshold", "cpu_idle")
for line in tf.splitlines():
    if any(k in line for k in keys) and not line.strip().startswith("#"):
        print("  tf |", line.strip())

# %% [markdown]
# ## Worked example: the same deployment as `gcloud` commands (dry run)

# %%
if shutil.which("bash"):
    out = subprocess.run(["bash", str(CR / "deploy.sh")], capture_output=True, text=True,
                         env={**os.environ, "DRY_RUN": "1", "PROJECT_ID": os.environ.get("PROJECT_ID", "my-project")})
    print(out.stdout[:2500] or out.stderr[:2500])
else:
    print("bash not available: read deploy/gcp/cloud-run/deploy.sh")

# %% [markdown]
# ## Worked example: does the model fit, and how much KV is left?

# %%
plan = sizing.size("qwen2.5-1.5b-instruct", "L4", max_model_len=8192, typical_len=768)
print(plan.summary())

# %% [markdown]
# ## Exercise 6.1 — anatomy of a cold start
#
# After a scale to zero, the first request waits for three things:
#
# - the image pull,
# - the read of the weights,
# - the init of the engine (the profile run, the KV allocation, the CUDA-graph capture).
#
# Write `cold_start_s(image_gb, pull_gb_s, weights_gb, weights_gb_s, init_s)`. Then compare two sources for
# the weights. The inputs in the next cell are **assumptions**, not measurements. Replace them with your own
# values. Get the image size from the registry and the bandwidths from a test. Get the init time from the log
# line `init engine (profile, create kv cache, warmup model) took ...`.
#
# Then comes the part that breaks deployments. The **startup probe** must last longer than the part of the
# cold start that occurs *inside* the container. The image pull comes before the container starts. Thus
# only the weights and the engine init count against the probe (as in Kubernetes, verify for Cloud Run).
#
# Write `min_failure_threshold(weights_gb, weights_gb_s, init_s, period_s, margin)`. It returns the smallest
# `failure_threshold` whose $\mathtt{failure\_threshold} \times \mathtt{period\_s}$ covers
# $\mathtt{margin} \times{}$ the time of the weight read and the engine init. The check reads the probe settings from the Terraform of this lab.

# %%
ASSUME = {"image_gb": 8.0,         # vllm/vllm-openai compressed size (assumption; verify in the registry)
          "pull_gb_s": 0.5,        # image pull throughput (assumption)
          "hf_gb_s": 0.1,          # download from the Hugging Face Hub (assumption)
          "gcs_gb_s": 1.0,         # read from a mounted Cloud Storage bucket (assumption)
          "init_s": 60.0}          # engine init incl. CUDA graphs (assumption; read your log)
WEIGHTS_GB = sizing.weight_bytes(sizing.load_config("qwen2.5-1.5b-instruct")) / 1e9
print(f"weights: {WEIGHTS_GB:.2f} GB (computed from config.json)")

# %% exercise
def cold_start_s(image_gb: float, pull_gb_s: float, weights_gb: float, weights_gb_s: float, init_s: float) -> float:
    ### BEGIN SOLUTION
    return image_gb / pull_gb_s + weights_gb / weights_gb_s + init_s
    ### END SOLUTION

def min_failure_threshold(weights_gb: float, weights_gb_s: float, init_s: float, period_s: float,
                          margin: float = 1.5) -> int:
    ### BEGIN SOLUTION
    return math.ceil(margin * (weights_gb / weights_gb_s + init_s) / period_s)
    ### END SOLUTION

# %% check
assert cold_start_s(8, 0.5, 3, 0.1, 60) == 16 + 30 + 60
hf = cold_start_s(ASSUME["image_gb"], ASSUME["pull_gb_s"], WEIGHTS_GB, ASSUME["hf_gb_s"], ASSUME["init_s"])
gcs = cold_start_s(ASSUME["image_gb"], ASSUME["pull_gb_s"], WEIGHTS_GB, ASSUME["gcs_gb_s"], ASSUME["init_s"])
assert gcs < hf
assert min_failure_threshold(3, 0.1, 60, 10, 1.0) == 9 and min_failure_threshold(3, 0.1, 60, 10) == 14
period = int(re.search(r"period_seconds\s*=\s*(\d+)", tf).group(1))
tf_threshold = int(re.search(r'variable "startup_failure_threshold"[^}]*default\s*=\s*(\d+)',
                             (CR / "terraform" / "variables.tf").read_text(), re.S).group(1))
need_hf = min_failure_threshold(WEIGHTS_GB, ASSUME["hf_gb_s"], ASSUME["init_s"], period)
slow_hf = min_failure_threshold(15.0, 0.05, 120, period)     # an 8B model on a slow day
print(f"   startup probe in the Terraform: {tf_threshold} x {period} s = {tf_threshold * period} s; this model from "
      f"the Hub needs >= {need_hf} failures; an 8B download at 50 MB/s would need {slow_hf}")
assert tf_threshold >= need_hf
print(f"✅ cold start under these assumptions: {hf:.0f} s with Hugging Face, {gcs:.0f} s with weights in GCS — "
      f"after that, image pull and engine init dominate; the probe budget covers weights + init with room")

# %% [markdown]
# ## Exercise 6.2 — choose Cloud Run `concurrency` from a measurement
#
# One instance is one engine. On the L4 profile, do a sweep of the number of simultaneous requests in the
# instance (closed loop, with the users spread over one second). Keep the largest level at which at
# least 90% of the requests meet the SLO. Write `pick_concurrency(summaries, min_attainment)` over a dict
# `{concurrency: Summary}`.

# %%
SLO_CR = SLO(ttft_ms=1000, tpot_ms=25)
summaries = {}
with FakeServer("l4-qwen2.5-1.5b") as url:
    for c in (1, 8, 32, 64, 128):
        reqs = random_requests(max(8, 2 * c), Lengths.fixed(256), Lengths.uniform(32, 96), seed=c)
        s = run_closed_loop(url, reqs, c, ramp_s=1.0).summary(SLO_CR)
        summaries[c] = s
        print(f"[SIMULATED L4] {c:>3} in flight: {s.output_throughput:7.0f} tok/s, TTFT p99 {s.ttft.p[99]:6.0f} ms, "
              f"TPOT p99 {s.tpot.p[99]:5.1f} ms, SLO met {s.slo_attainment:4.0%}")

# %% exercise
def pick_concurrency(summaries: dict, min_attainment: float = 0.9):
    ### BEGIN SOLUTION
    ok = [c for c, s in summaries.items() if s.slo_attainment >= min_attainment]
    return max(ok) if ok else None
    ### END SOLUTION

# %% check
def S(attainment):          # a minimal Summary: only slo_attainment matters here
    st = Stat(1, 1.0, 1.0, 0.0, {99: 1.0})
    return Summary(1, 0, 1.0, 1, 1, 1.0, 1.0, 1.0, attainment, attainment, st, st, st, st)
assert pick_concurrency({1: S(1.0), 8: S(0.95), 32: S(0.91), 64: S(0.4)}) == 32
assert pick_concurrency({1: S(1.0), 8: S(0.9), 32: S(0.89)}) == 8                  # exactly 0.9 counts
assert pick_concurrency({64: S(0.95), 1: S(1.0), 8: S(0.5), 32: S(0.2)}) == 64      # noisy, unsorted: largest level that meets it
assert pick_concurrency({1: S(0.5)}) is None and pick_concurrency(summaries, 1.01) is None
chosen = pick_concurrency(summaries)
print(f"✅ set concurrency = {chosen} (Terraform var.concurrency / gcloud --concurrency); beyond it Cloud Run "
      f"should add an instance rather than slow this one down")

# %% [markdown]
# ## Exercise 6.3 — cost per million output tokens
#
# The cost per token is the price of the instance divided by the tokens that it makes. Write
# `cost_per_million(price_per_hour, tokens_per_s)`. The price is an **assumption** for an L4 instance with
# 8 vCPU and 32 GiB on Cloud Run. Look at the pricing page (verify). The throughput is the measured
# throughput (here: simulated) at the concurrency that you selected.

# %%
PRICE_PER_HOUR = 1.00      # USD per L4 instance-hour on Cloud Run, incl. vCPU and memory (assumption; verify)

# %% exercise
def cost_per_million(price_per_hour: float, tokens_per_s: float) -> float:
    ### BEGIN SOLUTION
    return price_per_hour / 3600 / tokens_per_s * 1e6
    ### END SOLUTION

# %% check
assert abs(cost_per_million(3.6, 1000) - 1.0) < 1e-9
tps = summaries[chosen].output_throughput
print(f"✅ at {tps:.0f} output tok/s: ${cost_per_million(PRICE_PER_HOUR, tps):.2f} per million output tokens "
      f"(assumed ${PRICE_PER_HOUR:.2f}/h, simulated throughput); at 1 in flight: "
      f"${cost_per_million(PRICE_PER_HOUR, summaries[1].output_throughput):.2f}")

# %% [markdown]
# ## Exercise 6.4 — scale to zero or stay warm?
#
# Traffic comes in bursts: `busy_hours_per_day` of load in `bursts_per_day` separate bursts. With
# `min_instances = 0`, you pay for the busy hours. You also pay for an idle tail after each burst, before
# Cloud Run stops the instance (`idle_tail_h`, verify the current behaviour of Cloud Run). Each burst also
# starts with a cold start. With `min_instances = 1`, you pay for 24 hours a day, and you never wait.
#
# Write `monthly_costs(price_per_hour, busy_hours_per_day, bursts_per_day, idle_tail_h=0.25, days=30)`. It
# returns `(scale_to_zero_cost, always_warm_cost)` for one instance.

# %% exercise
def monthly_costs(price_per_hour: float, busy_hours_per_day: float, bursts_per_day: int,
                  idle_tail_h: float = 0.25, days: int = 30) -> tuple:
    ### BEGIN SOLUTION
    scale_to_zero = price_per_hour * (busy_hours_per_day + bursts_per_day * idle_tail_h) * days
    always_warm = price_per_hour * 24 * days
    return scale_to_zero, always_warm
    ### END SOLUTION

# %% check
assert monthly_costs(1.0, 4, 4) == (150.0, 720.0)
s0, warm = monthly_costs(PRICE_PER_HOUR, busy_hours_per_day=3, bursts_per_day=6)
print(f"✅ 3 busy hours/day in 6 bursts: ${s0:,.0f}/month scaling to zero (6 cold starts of ~{gcs:.0f} s a day) "
      f"vs ${warm:,.0f}/month always warm")

# %% [markdown]
# ## Exercise 6.5 — Cloud Run concurrency above the engine's batch cap
#
# The `concurrency` of Cloud Run and the `--max-num-seqs` of vLLM are two different limits. If Cloud Run
# lets 64 requests into an instance whose engine runs at most 16 at a time, 48 requests wait *inside vLLM*.
# Cloud Run sees a busy instance, not a queue. Predict the wait.
#
# The model is a closed loop of `concurrency` users against a batch limit `max_num_seqs`. Each admitted request
# takes `service_s` (its E2E at a full batch). Thus the engine completes
# $\mathtt{max\_num\_seqs} / \mathtt{service\_s}$ requests per second. By Little's law, each request spends
#
# $$
# \frac{\mathtt{concurrency} \times \mathtt{service\_s}}{\mathtt{max\_num\_seqs}}
# $$
#
# in the instance. The part after its own `service_s` is time in the queue.
#
# Write `queued_wait_s(concurrency, max_num_seqs, service_s)`. The check measures `service_s` with 16 users.
# Then it runs 64 users and compares the TTFT.

# %% exercise
def queued_wait_s(concurrency: int, max_num_seqs: int, service_s: float) -> float:
    ### BEGIN SOLUTION
    return max(0.0, concurrency / max_num_seqs - 1) * service_s
    ### END SOLUTION

# %% check
assert queued_wait_s(16, 16, 0.6) == 0.0 and math.isclose(queued_wait_s(64, 16, 0.6), 1.8)
with FakeServer("l4-qwen2.5-1.5b", EngineConfig(max_num_seqs=16)) as url:
    full = run_closed_loop(url, random_requests(48, Lengths.fixed(256), Lengths.fixed(32), seed=41), 16,
                           ramp_s=0.5).summary()
    over = run_closed_loop(url, random_requests(256, Lengths.fixed(256), Lengths.fixed(32), seed=42), 64,
                           ramp_s=0.5).summary()
service = full.e2el.mean / 1000
pred = queued_wait_s(64, 16, service) + full.ttft.mean / 1000
print(f"[SIMULATED L4] max_num_seqs 16: service {service:.2f} s at 16 users; at 64 users predicted TTFT "
      f"{pred:.2f} s, measured mean {over.ttft.mean / 1000:.2f} s (p99 {over.ttft.p[99] / 1000:.2f} s)")
assert abs(over.ttft.mean / 1000 / pred - 1) < 0.3
print("✅ concurrency above max_num_seqs becomes a queue inside the engine: set Cloud Run's concurrency to the batch "
      "that meets the SLO, so the extra load starts a new instance instead")

# %% [markdown]
# ## Deploy for real (T3, opt-in)
#
# You must have an authenticated `gcloud`, `PROJECT_ID` set, a paid billing account and L4 quota for Cloud
# Run in the region. The Terraform path: `cd deploy/gcp/cloud-run/terraform && terraform apply`. The script
# path: the next cell, with `SERVELAB_DEPLOY=1`. After that, measure the service with the same code as on
# your computer: `gcloud run services proxy vllm-l4 --port 8080`, then `SERVELAB_URL=http://127.0.0.1:8080`
# and notebooks 02-05. When you finish, delete the service: `./deploy.sh delete` or `terraform destroy`.

# %%
if os.environ.get("SERVELAB_DEPLOY") == "1" and env.has_gcloud() and os.environ.get("PROJECT_ID"):
    subprocess.run(["bash", str(CR / "deploy.sh")], check=True)
else:
    print("Not deploying (set SERVELAB_DEPLOY=1 and PROJECT_ID, with gcloud installed). Plan above is T0.")

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "Each Cloud Run instance is one vLLM on one L4. First, we calculated the size of the
# model (a 1.5B model in bf16 leaves ~17 GiB of KV on the L4). Then we measured one instance: in this
# notebook, the simulated L4 profile, and on GCP, the real service. With 32 requests in flight, we still met
# our SLO, and with 64 we did not. Thus `concurrency` is 32. Cloud Run adds an instance and does not let
# one engine make all requests slower.
#
# "We scale to zero because the traffic comes in bursts and the bill is per instance-second. The price is a
# cold start of a minute or two. When the weights come from Cloud Storage, the image and the engine init
# are the largest parts of that time. If the first request of a burst cannot wait, we keep
# `min_instances = 1` and pay for 24 hours. The cost per million tokens is the instance price divided by
# the measured throughput. Thus good batching is also what makes the cost low."
#
# **Drill 1.** *Why not set Cloud Run concurrency to 1,000 and let vLLM batch?* vLLM will batch the
# requests. But when the batch is larger than the SLO-limited batch of the engine, each request becomes
# slower or waits in a queue inside the instance. Cloud Run never scales out, because the instance never looks full.
#
# **Drill 2.** *The first request after lunch takes 90 seconds. Three corrections?* Set `min_instances = 1`
# (pay to stay warm). Get the weights from Cloud Storage instead of the Hub. Make the engine start faster
# with a smaller image and with `--enforce-eager`, which skips CUDA-graph capture at some cost in decode
# speed.
#
# **Drill 3.** *Cloud Run or GKE for this?* Use Cloud Run for a service that scales to zero and has traffic
# in bursts, with one GPU per instance and no cluster to operate. Use GKE when you must have multi-GPU nodes,
# engine-aware routing and autoscaling on queue depth or KV usage (layer 05), Spot capacity or
# reservations.
