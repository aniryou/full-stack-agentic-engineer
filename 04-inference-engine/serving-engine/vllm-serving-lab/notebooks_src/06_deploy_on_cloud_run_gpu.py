# %% [markdown]
# # 06 · Deploy on Cloud Run GPU: cold starts, concurrency and cost, planned before you apply
#
# **Tier:** T3 walkthrough (GCP), fully plannable at T0: the Terraform and the `gcloud` script are
# read and dry-run here, the engine is the fake vLLM with an **L4 profile** (results **simulated**),
# and every price or bandwidth is an explicit *assumption* you replace. With `gcloud`, a project and
# `SERVELAB_DEPLOY=1`, the last cell deploys for real; then set `SERVELAB_URL` and re-run notebook 02.
#
# ## The one-minute version
#
# A Cloud Run GPU service is a pool of identical instances, each one engine on one L4 (24 GB). Three
# settings decide latency and cost:
#
# * **`concurrency`** — how many requests Cloud Run sends to one instance. It should equal the batch
#   the engine can serve *within the SLO*; more queues inside the engine (TPOT and TTFT suffer),
#   less scales out earlier than necessary (cost).
# * **`min_instances`** — 0 means scale to zero: no cost while idle, a **cold start** (image pull +
#   weights + engine init) for the first request after idle. 1 means always warm, always billing.
# * **`max_instances`** — a cap on GPUs, on the bill, and on the load you can absorb.
#
# Concepts: PRIMER §12 "Engines and where to run them" ([`PRIMER.md`](../../PRIMER.md)); the deploy
# assets are in `deploy/gcp/cloud-run/` (Terraform and `gcloud`). Prices: `COMPUTE.md` at the repo root.

# %%
import os, shutil, subprocess
from pathlib import Path
import servelab
from servelab import env, sizing
from servelab.bench import SLO, Lengths, random_requests, run_closed_loop
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
# After scale-to-zero the first request waits for: pulling the image, reading the weights, and the
# engine's init (profiling run, KV allocation, CUDA-graph capture). Write
# `cold_start_s(image_gb, pull_gb_s, weights_gb, weights_gb_s, init_s)`, then compare two sources for
# the weights. The inputs below are **assumptions**, not measurements — replace them with your own
# (image size from the registry, bandwidths from a test, init time from the log line
# `init engine (profile, create kv cache, warmup model) took ...`).

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

# %% check
assert cold_start_s(8, 0.5, 3, 0.1, 60) == 16 + 30 + 60
hf = cold_start_s(ASSUME["image_gb"], ASSUME["pull_gb_s"], WEIGHTS_GB, ASSUME["hf_gb_s"], ASSUME["init_s"])
gcs = cold_start_s(ASSUME["image_gb"], ASSUME["pull_gb_s"], WEIGHTS_GB, ASSUME["gcs_gb_s"], ASSUME["init_s"])
assert gcs < hf
print(f"✅ cold start under these assumptions: {hf:.0f} s with Hugging Face, {gcs:.0f} s with weights in GCS — "
      f"after that, image pull and engine init dominate")

# %% [markdown]
# ## Exercise 6.2 — choose Cloud Run `concurrency` from a measurement
#
# One instance = one engine. Sweep the number of simultaneous requests it holds (closed loop,
# users staggered over one second) on the L4 profile and keep the largest level at which at least
# 90% of requests meet the SLO. Write `pick_concurrency(summaries, min_attainment)` over a dict
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
chosen = pick_concurrency(summaries)
assert chosen == max(c for c, s in summaries.items() if s.slo_attainment >= 0.9)
assert pick_concurrency(summaries, 1.01) is None
print(f"✅ set concurrency = {chosen} (Terraform var.concurrency / gcloud --concurrency); beyond it Cloud Run "
      f"should add an instance rather than slow this one down")

# %% [markdown]
# ## Exercise 6.3 — cost per million output tokens
#
# Cost per token is the instance's price divided by the tokens it produces. Write
# `cost_per_million(price_per_hour, tokens_per_s)`. The price is an **assumption** (an L4 instance
# with 8 vCPU and 32 GiB on Cloud Run; check the pricing page — verify); the throughput is the one
# measured (here: simulated) at the concurrency you chose.

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
# Traffic arrives in bursts: `busy_hours_per_day` of load in `bursts_per_day` separate bursts. With
# `min_instances = 0` you pay for busy hours plus an idle tail after each burst before the instance
# is shut down (`idle_tail_h`; verify Cloud Run's current behaviour) — and every burst starts with a
# cold start. With `min_instances = 1` you pay for 24 hours a day and never wait. Write
# `monthly_costs(price_per_hour, busy_hours_per_day, bursts_per_day, idle_tail_h=0.25, days=30)`
# returning `(scale_to_zero_cost, always_warm_cost)` for one instance.

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
# ## Deploy for real (T3, opt-in)
#
# Needs `gcloud` authenticated, `PROJECT_ID` set, a paid billing account and L4 quota for Cloud Run
# in the region. Terraform path: `cd deploy/gcp/cloud-run/terraform && terraform apply`. Script path:
# the cell below, with `SERVELAB_DEPLOY=1`. Afterwards measure it with the same code as locally:
# `gcloud run services proxy vllm-l4 --port 8080`, then `SERVELAB_URL=http://127.0.0.1:8080` and
# notebooks 02-05. Delete it when you are done: `./deploy.sh delete` or `terraform destroy`.

# %%
if os.environ.get("SERVELAB_DEPLOY") == "1" and env.has_gcloud() and os.environ.get("PROJECT_ID"):
    subprocess.run(["bash", str(CR / "deploy.sh")], check=True)
else:
    print("Not deploying (set SERVELAB_DEPLOY=1 and PROJECT_ID, with gcloud installed). Plan above is T0.")

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "Each Cloud Run instance is one vLLM on one L4. We sized the model first (a
# 1.5B model in bf16 leaves ~17 GiB of KV on the L4), then measured one instance (in this notebook,
# the simulated L4 profile; on GCP, the real service): 32 requests in flight still met our SLO and
# 64 did not, so `concurrency` is 32 — Cloud Run adds an instance
# rather than letting one engine slow everyone down. We scale to zero because traffic is bursty and
# the bill is per instance-second; the price is a cold start of a minute or two, dominated by the
# image and engine init once the weights come from Cloud Storage. If the first request of a burst
# cannot wait, we keep `min_instances = 1` and pay for 24 hours. Cost per million tokens is the
# instance price over measured throughput, so batching well is also what makes it cheap."
#
# **Drill 1.** *Why not set Cloud Run concurrency to 1,000 and let vLLM batch?* — vLLM will batch,
# but past the engine's SLO-limited batch every request slows down or queues inside the instance,
# and Cloud Run never scales out because the instance never looks full.
#
# **Drill 2.** *The first request after lunch takes 90 seconds. Three fixes?* — `min_instances = 1`
# (pay to stay warm), weights from Cloud Storage instead of the Hub, and a faster engine start
# (smaller image; `--enforce-eager` skips CUDA-graph capture at some cost in decode speed).
#
# **Drill 3.** *Cloud Run or GKE for this?* — Cloud Run for bursty, scale-to-zero serving with one
# GPU per instance and no cluster to run; GKE when you need multi-GPU nodes, engine-aware routing
# and autoscaling on queue depth or KV usage (layer 05), Spot capacity or reservations.
