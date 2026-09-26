# %% [markdown]
# # 02 · Serve and measure: TTFT, ITL, TPOT, throughput and goodput — defined, measured, cross-checked
#
# **Tier:** T0 by default — a fake vLLM starts in-process and every number it produces is labelled
# **simulated**. T1/T3: set `SERVELAB_URL` to a real OpenAI-compatible server (your `vllm serve`, a
# Colab T4, a Cloud Run URL) and the same cells measure it.
#
# ## The one-minute version
#
# Serving numbers are ratios with definitions, measured at the client from the stream of
# server-sent events (the definitions of `vllm bench serve`):
#
# | Metric | Definition | What it is about |
# |---|---|---|
# | **TTFT** | first token chunk − send time | queueing + prefill (+ network) |
# | **ITL** | gaps between consecutive chunks | one decode step, stalls when a prefill shares the step |
# | **TPOT** | (E2E − TTFT) / (output tokens − 1), per request | the per-token pace a user feels |
# | **E2E** | last chunk − send time | the whole request |
# | **throughput** | tokens (or requests) / wall time of the run | capacity |
# | **goodput** | requests meeting *every* SLO / wall time | capacity you can sell |
#
# The engine's own `/metrics` gives the server-side view: queue depth, batch size, KV usage and
# latency *histograms*, whose percentiles are interpolations inside buckets. How requests are sent
# matters as much as how they are timed: an **open loop** exposes overload, a **closed loop** hides
# it. Concepts: PRIMER §11 "Measuring an engine" ([`../PRIMER.md`](../PRIMER.md)).

# %%
import json, math, threading, time, urllib.request
from servelab import env, metrics as M
from servelab.bench import (SLO, Lengths, random_requests, run_closed_loop, run_open_loop, run_sync, summarize)
from servelab.bench.client import RequestResult, stream_request
from servelab.bench.runner import _session
from servelab.fake_engine import EngineConfig
from servelab.fakeserver import FakeServer

print(env.describe())
target = env.connect("t4-qwen2.5-0.5b")      # SERVELAB_URL -> real server; otherwise the fake one (simulated)
URL, H = target.url, target.headers
print(target)
MODEL = json.loads(urllib.request.urlopen(urllib.request.Request(URL + "/v1/models", headers=H or {})).read())["data"][0]["id"]
LABEL = "SIMULATED" if target.simulated else "MEASURED"
print("model:", MODEL, "|", LABEL)

# %% [markdown]
# ## Worked example: what a streaming response looks like on the wire
#
# One `/v1/completions` request with `"stream": true`: a sequence of `data: {json}` events, one per
# engine step that produced text, then a usage event (because of `stream_options.include_usage`)
# and `data: [DONE]`.

# %%
body = {"model": MODEL, "prompt": "Explain paged attention in one sentence.", "max_tokens": 6, "stream": True,
        "stream_options": {"include_usage": True}, "temperature": 0}
req = urllib.request.Request(URL + "/v1/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json", **(H or {})})
events = [l for l in urllib.request.urlopen(req, timeout=120).read().decode().split("\n\n") if l]
for e in events[:2] + ["..."] + events[-2:]:
    print(e[:160])

# %% [markdown]
# ## Worked example: one request, timed chunk by chunk

# %%
async def one(prompt_tokens=512, max_tokens=32):
    from servelab.bench import Request
    from servelab.textgen import synthetic_text
    async with _session() as http:
        return await stream_request(http, URL, MODEL, Request(prompt=synthetic_text(prompt_tokens, 99),
                                                              max_tokens=max_tokens), headers=H)

r = run_sync(one())
print(f"[{LABEL}] TTFT {r.ttft * 1e3:.1f} ms | ITL first gaps (ms): {[round(g * 1e3, 1) for g in r.itl[:6]]} ...")
print(f"output tokens {r.output_tokens}, E2E {r.latency * 1e3:.1f} ms, TPOT {r.tpot * 1e3:.2f} ms, "
      f"prompt tokens (server-counted) {r.prompt_tokens}")

# %% [markdown]
# ## Worked example: an open-loop run, with the engine's view of the same window
#
# 40 requests at 6 req/s (Poisson arrivals), 512-token prompts, 64 output tokens. We scrape
# `/metrics` before and after, so the engine's counters describe exactly this run.

# %%
reqs = random_requests(40, Lengths.fixed(512), Lengths.fixed(64), seed=1)
before = M.scrape(URL, headers=H)
run = run_open_loop(URL, reqs, rate=6.0, seed=1, headers=H)
after = M.scrape(URL, headers=H)
SLO_CHAT = SLO(ttft_ms=300, tpot_ms=30)
print(run.report(SLO_CHAT))
print(M.snapshot(after, before).table())

# %% [markdown]
# ## Exercise 2.1 — the per-request definitions
#
# Given the send time, the arrival time of each token chunk and the output token count reported in
# `usage`, return a dict with `ttft`, `itl` (list), `e2e` and `tpot` (seconds). `tpot` is
# `(e2e - ttft) / (output_tokens - 1)`, and `nan` for single-token outputs.

# %% exercise
def request_metrics(send_time: float, chunk_times: list, output_tokens: int) -> dict:
    ### BEGIN SOLUTION
    ttft = chunk_times[0] - send_time
    e2e = chunk_times[-1] - send_time
    itl = [b - a for a, b in zip(chunk_times, chunk_times[1:])]
    tpot = (e2e - ttft) / (output_tokens - 1) if output_tokens > 1 else math.nan
    return {"ttft": ttft, "itl": itl, "e2e": e2e, "tpot": tpot}
    ### END SOLUTION

# %% check
hand = request_metrics(10.0, [10.2, 10.25, 10.3, 10.4], 4)
assert math.isclose(hand["ttft"], 0.2) and math.isclose(hand["e2e"], 0.4) and math.isclose(hand["tpot"], 0.2 / 3)
assert [round(x, 3) for x in hand["itl"]] == [0.05, 0.05, 0.1]
assert math.isnan(request_metrics(0.0, [0.1], 1)["tpot"])
x = run.results[0]
mine = request_metrics(x.start, x.chunk_times, x.output_tokens)
assert math.isclose(mine["ttft"], x.ttft) and math.isclose(mine["tpot"], x.tpot) and len(mine["itl"]) == len(x.itl)
print("✅ request_metrics matches the benchmark's definitions (and vLLM's)")

# %% [markdown]
# ## Exercise 2.2 — goodput
#
# Goodput counts only requests that met **every** SLO, per second of the run. Write
# `goodput(results, duration_s, ttft_ms, tpot_ms)` using vLLM's rule: a request is good when
# `ttft <= target` and `tpot <= target`, where a single-token output counts as TPOT 0; failed
# requests are never good. Then compare with plain throughput: the gap is the capacity you
# cannot sell at this SLO.

# %% exercise
def goodput(results: list, duration_s: float, ttft_ms: float, tpot_ms: float) -> float:
    ### BEGIN SOLUTION
    good = 0
    for r in results:
        if not r.ok:
            continue
        tpot = r.tpot if r.output_tokens > 1 else 0.0
        if r.ttft * 1000 <= ttft_ms and tpot * 1000 <= tpot_ms:
            good += 1
    return good / duration_s
    ### END SOLUTION

# %% check
fake = [RequestResult(ok=True, ttft=0.1, latency=0.1, output_tokens=1),                   # TPOT counts 0
        RequestResult(ok=True, ttft=0.1, latency=0.1 + 0.05 * 9, output_tokens=10),       # TPOT 50 ms
        RequestResult(ok=False)]
assert goodput(fake, 2.0, ttft_ms=200, tpot_ms=40) == 0.5 and goodput(fake, 2.0, ttft_ms=200, tpot_ms=60) == 1.0
for t in (100, 300, 1000):
    assert math.isclose(goodput(run.results, run.duration_s, t, 30), summarize(run.results, run.duration_s,
                                                                                 SLO(ttft_ms=t, tpot_ms=30)).goodput)
s = run.summary(SLO_CHAT)
print(f"✅ [{LABEL}] throughput {s.request_throughput:.2f} req/s, goodput at {SLO_CHAT}: {s.goodput:.2f} req/s")

# %% [markdown]
# ## The server's view: exact means, interpolated percentiles
#
# A Prometheus histogram stores *counts per bucket*, plus a sum and a count. The mean
# (`_sum / _count`) is exact; a percentile is found by locating the bucket that holds the rank
# and interpolating linearly inside it. vLLM's first queue-time bucket is 0-0.3 s, so a run with
# no queueing at all reports a "p50 queue time" of ~150 ms — pure interpolation.

# %%
w = M.delta(after, before)
for name, metric in (("TTFT", M.TTFT), ("queue", M.QUEUE), ("ITL", M.ITL)):
    h = w.histogram(metric)
    print(f"{name:6s} server mean {h.mean * 1e3:7.1f} ms | server p50 {h.quantile(0.5) * 1e3:7.1f} ms "
          f"| buckets around it: {[le for le, _ in h.buckets[:4]]} ...")
print(f"client TTFT mean {run.summary().ttft.mean:.1f} ms (includes HTTP, tokenization and the network)")

# %% [markdown]
# ## Exercise 2.3 — `histogram_quantile`, as PromQL computes it
#
# Implement it for `buckets = [(upper_bound, cumulative_count), ...]` ending with `+Inf`:
# `rank = q × total`; take the first bucket whose cumulative count is `>= rank`; interpolate
# linearly from the previous upper bound (0 for the first bucket) to this one. If the rank lands
# in the `+Inf` bucket, return the largest finite bound. No observations: `nan`.

# %% exercise
def histogram_quantile(q: float, buckets: list) -> float:
    ### BEGIN SOLUTION
    b = sorted(buckets)
    total = b[-1][1]
    if total == 0:
        return math.nan
    rank = q * total
    for i, (le, c) in enumerate(b):
        if c >= rank:
            if math.isinf(le):
                return b[i - 1][0]
            lo_le, lo_c = (b[i - 1] if i else (0.0, 0.0))
            return lo_le + (le - lo_le) * (rank - lo_c) / (c - lo_c)
    return b[-2][0]
    ### END SOLUTION

# %% check
inf = math.inf
assert math.isclose(histogram_quantile(0.5, [(0.1, 10), (0.25, 30), (0.5, 40), (inf, 40)]), 0.175)
assert math.isclose(histogram_quantile(0.5, [(1.0, 10), (inf, 10)]), 0.5)            # starts at 0
assert histogram_quantile(0.99, [(1.0, 0), (2.0, 1), (inf, 100)]) == 2.0              # +Inf bucket
assert math.isnan(histogram_quantile(0.5, [(1.0, 0), (inf, 0)]))
live = w.histogram(M.TTFT).buckets
for q in (0.5, 0.9, 0.99):
    assert math.isclose(histogram_quantile(q, live), M.histogram_quantile(q, live))
print("✅ histogram_quantile matches PromQL's algorithm; server TTFT p99 =",
      f"{histogram_quantile(0.99, live) * 1e3:.1f} ms (an interpolation)")

# %% [markdown]
# ## Open loop versus closed loop, on a server you can overload
#
# To see overload cheaply we start a second fake server whose batch is capped at 4 sequences
# (`max_num_seqs=4`) — *simulated* at every tier; on a GPU, repeat it with
# `vllm serve ... --max-num-seqs 4`. First a closed loop: 4 users, each sends the next request when
# the previous answer is complete.

# %%
small = FakeServer("t4-qwen2.5-0.5b", EngineConfig(max_num_seqs=4))
SMALL = small.start()
work = lambda seed: random_requests(40, Lengths.fixed(256), Lengths.fixed(64), seed=seed)  # noqa: E731
closed = run_closed_loop(SMALL, work(2), concurrency=4)
cs = closed.summary()
print(f"[SIMULATED] closed loop, 4 users: {cs.request_throughput:.2f} req/s, TTFT p99 {cs.ttft.p[99]:.0f} ms, "
      f"mean E2E {cs.e2el.mean:.0f} ms")

# %% [markdown]
# ## Exercise 2.4 — capacity from Little's law, and which loop tells the truth
#
# Little's law says requests in the system = arrival rate × time each spends there. With at most
# `max_num_seqs` requests running, each taking `mean_e2e_s`, the engine completes at most
# `capacity_rps = max_num_seqs / mean_e2e_s` requests per second. Implement it, then predict: if we
# now send the *same* requests **open-loop at twice that capacity**, which run shows the larger TTFT
# p99 — `"open"` or `"closed"`? Set `worse_tail` accordingly.

# %% exercise
def capacity_rps(max_num_seqs: int, mean_e2e_s: float) -> float:
    ### BEGIN SOLUTION
    return max_num_seqs / mean_e2e_s
    ### END SOLUTION

### BEGIN SOLUTION
worse_tail = "open"
### END SOLUTION

# %% check
cap = capacity_rps(4, cs.e2el.mean / 1000)
assert abs(cap / cs.request_throughput - 1) < 0.25, (cap, cs.request_throughput)
opened = run_open_loop(SMALL, work(2), rate=2 * cap, seed=3)
os_ = opened.summary()
print(f"[SIMULATED] open loop at {2 * cap:.1f} req/s: TTFT p99 {os_.ttft.p[99]:.0f} ms vs closed {cs.ttft.p[99]:.0f} ms; "
      f"throughput {os_.request_throughput:.2f} vs {cs.request_throughput:.2f} req/s")
assert worse_tail == ("open" if os_.ttft.p[99] > cs.ttft.p[99] else "closed")
assert worse_tail == "open"
small.stop()
print("✅ same server, same capacity: the closed loop slowed its own arrivals and hid the queue")

# %% [markdown]
# ## Exercise 2.5 — Little's law against the engine's gauges
#
# While a run is in flight, a background thread samples `vllm:num_requests_running +
# vllm:num_requests_waiting` from `/metrics`. Write `inflight(arrival_rate, mean_latency_s)` and
# check that the prediction from *client-side* numbers matches the *server-side* average.

# %%
samples, stop = [], threading.Event()

def sampler():
    while not stop.is_set():
        s = M.scrape(URL, headers=H)
        samples.append(s.value(M.RUNNING, 0.0) + s.value(M.WAITING, 0.0))
        time.sleep(0.05)

th = threading.Thread(target=sampler, daemon=True)
th.start()
ll = run_open_loop(URL, random_requests(40, Lengths.fixed(256), Lengths.fixed(96), seed=7), rate=8.0, seed=7, headers=H)
stop.set()
th.join()
ls = ll.summary()
print(f"[{LABEL}] {len(samples)} samples, mean in flight {sum(samples) / len(samples):.2f}; "
      f"{ls.request_throughput:.2f} req/s x {ls.e2el.mean / 1000:.3f} s mean E2E")

# %% exercise
def inflight(arrival_rate: float, mean_latency_s: float) -> float:
    ### BEGIN SOLUTION
    return arrival_rate * mean_latency_s
    ### END SOLUTION

# %% check
predicted = inflight(ls.request_throughput, ls.e2el.mean / 1000)
observed = sum(samples) / len(samples)
assert abs(predicted / observed - 1) < 0.35, (predicted, observed)
print(f"✅ Little's law: predicted {predicted:.2f} in flight, engine gauges averaged {observed:.2f}")

# %%
target.stop()

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "We measure at the client, from the stream: TTFT to the first token chunk, ITL
# between chunks, TPOT per request as (E2E − TTFT)/(n − 1), throughput over the run's wall time —
# the definitions of `vllm bench serve`, so our numbers compare with anyone's. We report goodput:
# requests that met both the TTFT and the TPOT target per second. We drive the engine open-loop at
# a rate, because a closed loop slows down with the server and hides the queue. We cross-check
# with the engine's `/metrics`: the running and waiting gauges must agree with Little's law from the
# client side, and we treat histogram percentiles as bucket interpolations — the exact numbers
# there are the means."
#
# **Drill 1.** *Throughput went up 20% and p99 TTFT doubled. Is that better?* — Only if goodput
# went up; at a fixed SLO, more requests missing the TTFT target can mean less sellable capacity.
#
# **Drill 2.** *The Grafana panel says p50 queue time 150 ms, but users see no delay. Who is right?*
# — Probably the users: vLLM's first queue-time bucket is 0-0.3 s, so a queue of ~0 interpolates to
# half the bucket. Look at `_sum/_count` (the exact mean) or `vllm:num_requests_waiting`.
#
# **Drill 3.** *Why is ITL not the same as TPOT?* — ITL is per chunk; one chunk can carry several
# tokens (speculative decoding, `--stream-interval`), and TPOT averages the whole decode per token.
