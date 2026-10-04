# %% [markdown]
# # 04 · The local stack with llm-d, and real vLLM
#
# **Tier:** T0 walkthrough on any machine. It reads the deploy files, does a dry run of the scripts
# and runs the in-process stack. **T0 + Docker** applies when the compose or kind stack of
# `deploy/local` / `deploy/kind` runs. It still uses only the CPU, because the model servers are
# simulators. **T1/T2** applies when you point it at real vLLM replicas on a GPU (`deploy/any-gpu`,
# environment variable `IGW_BACKENDS`). Each measurement cell tells which of these it ran on.
#
# ## The one-minute version
#
# The decision logic of your exercises in notebooks 01–02 exists as production components:
#
# | piece | in-process (T0) | docker compose (`deploy/local`) | kind (`deploy/kind`) | any GPU box (`deploy/any-gpu`) |
# |---|---|---|---|---|
# | model servers | `igwlab.fakebackend` | 3 × `llm-d-inference-sim` (or the fake) | 3 × `llm-d-inference-sim` pods | 2+ × **real vLLM** |
# | proxy | the lab router | the lab router | Envoy (sidecar), ext-proc to the EPP | the lab router |
# | endpoint picker | the lab router | the lab router | **llm-d EPP** (Helm, standalone chart) | the lab router |
# | pool definition | a Python list | router flags | `InferencePool` v1 (selector + target ports) | `IGW_BACKENDS` |
# | request classes | `RouterSettings.objectives` | `--objective` flags | `InferenceObjective` CRs | `--objective` flags |
#
# The EndpointPickerConfig is the same document everywhere. The kind Helm values contain the
# `default-weighted` preset of the lab with no change. But the lab router and the EPP do not run it in
# the same way, and this notebook makes you name the differences.
#
# The simulator has the same *per-request* latency formula and the same 16-token prefix cache as the
# fake backend. It does not have the same contention. The fake puts prefills in a queue behind each
# other, but the simulator does not. Thus cache-aware routing gives a smaller TTFT gain on the
# simulator.
#
# Background: [PRIMER §9 The Kubernetes-native stack, September 2026 and §10 Where to run it](../../PRIMER.md).

# %%
import json
import os
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

import yaml

LAB = Path.cwd().resolve()
while not (LAB / "igwlab").exists():
    LAB = LAB.parent
DEPLOY = LAB / "deploy"
tools = {t: bool(shutil.which(t)) for t in ("docker", "kind", "kubectl", "helm", "nvidia-smi", "vllm")}
print("tools on this machine:", tools)

# %% [markdown]
# ## Path 1: docker compose
#
# `deploy/local/docker-compose.yaml` has two profiles: `sim` (the upstream simulator image) and `fake`
# (the bundled fake backend, with no pull from a registry). The router is the lab router with the
# `default-weighted` preset. Prometheus scrapes everything.

# %%
compose = yaml.safe_load((DEPLOY / "local/docker-compose.yaml").read_text())
for name, svc in compose["services"].items():
    print(f"{name:<12} profiles={svc.get('profiles')} image={svc.get('image')}")
print("\nsimulator flags:", " ".join(compose["services"]["sim-a"]["command"]))


def sim_flags(args) -> dict:
    """`--flag value` / `--flag=value` / bare `--flag` -> {flag: value}."""
    flags, i = {}, 0
    while i < len(args):
        a = args[i]
        if a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            flags[k] = v
        elif a.startswith("--"):
            nxt = args[i + 1] if i + 1 < len(args) else None
            if nxt is not None and not nxt.startswith("--"):
                flags[a[2:]] = nxt
                i += 1
            else:
                flags[a[2:]] = "true"
        i += 1
    return flags


def seconds(go_duration: str) -> float:
    n, unit = re.fullmatch(r"([0-9.]+)(us|µs|ms|s)", go_duration).groups()
    return float(n) * {"us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.0}[unit]


deployment = next(yaml.safe_load_all((DEPLOY / "kind/sim-deployment.yaml").read_text()))   # first document
kind_args = deployment["spec"]["template"]["spec"]["containers"][0]["args"]
f = sim_flags(kind_args)
print("kind simulator:", {k: f[k] for k in ("prefill-overhead", "prefill-time-per-token", "inter-token-latency",
                                           "time-factor-under-load", "block-size", "max-num-seqs")})

# %% [markdown]
# The `per-token` latency calculator of the simulator is the per-request model of the fake backend:
#
# $$
# \text{TTFT} = \texttt{prefill-overhead} + (\text{prompt} - \text{cached}) \times \texttt{prefill-time-per-token}
# $$
#
# Each output token adds `inter-token-latency`. As the batch fills, the simulator increases the TTFT
# and the `inter-token-latency` by a factor of up to `time-factor-under-load`. Both engines cache
# prompts in **full 16-token blocks**. An engine can use a block again only if all 16 of its tokens match. The engines never
# cache a partial last block.
#
# ## Exercise 4.1 — predict the second turn
#
# Turn 1 of an agent had `prev_prompt_tokens` prompt tokens. Turn 2 sends all of them again to the
# same replica and adds more tokens, `prompt_tokens` in total. Predict how many tokens of turn 2 come
# from the prefix cache. Also predict the TTFT that the simulator (flags `f`) emulates for turn 2.
# Turn 2 runs alone: no queue, a batch of one. Return `(cached_tokens, ttft_seconds)`.

# %% exercise
def predict_turn2(prev_prompt_tokens: int, prompt_tokens: int, flags: dict, block: int = 16):
    ### BEGIN SOLUTION
    cached = (prev_prompt_tokens // block) * block          # only turn 1's FULL blocks were cached
    cached = min(cached, ((prompt_tokens - 1) // block) * block)   # and at least one token is always computed
    ttft = seconds(flags["prefill-overhead"]) + (prompt_tokens - cached) * seconds(flags["prefill-time-per-token"])
    return cached, ttft
    ### END SOLUTION

# %% check
from igwlab.bench import agentic_sessions, session_messages
from igwlab.fakebackend import EngineProfile
from igwlab.stack import LocalStack

s0 = agentic_sessions(n_sessions=1, turns=2, n_agents=1, system_words=700, tool_words=150, seed=4)[0]
with LocalStack(1, "round-robin") as st:                         # one fake replica, in process (emulated timing)
    def turn(replies):
        body = json.dumps({"model": "lab/llm", "messages": session_messages(s0, replies), "max_tokens": 8}).encode()
        req = urllib.request.Request(st.router_url + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    r1 = turn([])
    r2 = turn([r1["choices"][0]["message"]["content"]])
q, p_ = r1["usage"]["prompt_tokens"], r2["usage"]["prompt_tokens"]
got = r2["usage"]["prompt_tokens_details"]["cached_tokens"]
cached, ttft = predict_turn2(q, p_, f)
assert cached == got, (cached, got)
assert abs(ttft - EngineProfile().prefill_seconds(p_, got)) < 1e-12
print(f"turn 1: {q} tokens, turn 2: {p_} tokens -> {got} cached ({q - got} of turn 1's tokens sat in its partial last block)")
print(f"✅ predicted TTFT alone on the simulator: {ttft * 1e3:.1f} ms vs {predict_turn2(0, p_, f)[1] * 1e3:.1f} ms "
      "with a cold cache (flags from deploy/kind; the cached count was measured on the in-process fake)")

# %% [markdown]
# ## Path 2: kind + the real llm-d Router (standalone mode)
#
# `deploy/kind/up.sh` does these steps:
#
# 1. It creates a cluster.
# 2. It installs the two CRDs: InferencePool v1 from the Gateway API Inference Extension, and
#    InferenceObjective from llm-d-router.
# 3. It deploys three simulator pods.
# 4. It installs the `llm-d-router-standalone` Helm chart.
#
# The chart contains an EPP Deployment with an Envoy sidecar. The sidecar receives client traffic on
# :8081 and asks the EPP (ext-proc) where to send each request. The chart mounts
# `router.epp.pluginsCustomConfig[<pluginsConfigFile>]` as the `--config-file` of the EPP. With
# `DRY_RUN=1`, the script prints the steps and runs nothing:

# %%
out = subprocess.run(["bash", str(DEPLOY / "kind/up.sh")], env={"DRY_RUN": "1", "PATH": "/usr/bin:/bin"},
                     capture_output=True, text=True, check=True).stdout
print("\n".join(l for l in out.splitlines() if l.startswith(("==>", "+ "))))
values = yaml.safe_load((DEPLOY / "kind/router-values.yaml").read_text())
epp = values["router"]["epp"]
epp_config = yaml.safe_load(epp["pluginsCustomConfig"][epp["pluginsConfigFile"]])
print("\nthe EPP on kind runs:", [(p["pluginRef"], p.get("weight")) for p in epp_config["schedulingProfiles"][0]["plugins"]])
print("objectives created by the chart:", values["router"]["inferenceObjectives"])

# %% [markdown]
# ## Exercise 4.2 — same config, same behaviour?
#
# For each situation, compare the EPP on kind (llm-d-router v0.10.0) with the lab router. Give one of
# these answers:
#
# - `"same"`: the same plugins, weights and formulas. The two are different only on exact ties and
#   when the prefix index has no more space.
# - `"differs"`: both accept the config, but they behave differently during normal operation.
# - `"lab-rejects"`: the lab router refuses the config.
#
# The README of the lab lists its intentional differences. The modules also state them.
#
# | | situation |
# |---|---|
# | a | the kind values' config (`default-weighted`: prefix ×3, queue ×2, KV ×2) on single-engine pods, all scraped |
# | b | the same config plus `featureGates: [flowControl]`, with a `batch` objective at priority −10 and the pool saturated |
# | c | the `round-robin` preset (no scorers), 300 requests over 3 endpoints |
# | d | two scheduling profiles, `prefill` and `decode` (P/D disaggregation) |
# | e | `prefix-cache-affinity-filter` with `ttftSource: latencyPredictor` |
# | f | `queue-scorer` on pods that each expose two data-parallel ranks (`engine="0"`, `engine="1"`) |

# %% exercise
verdicts = {}          # {"a": ..., "b": ..., ...}
### BEGIN SOLUTION
verdicts = {"a": "same",           # same plugins, weights and formulas (ties: rotation here, random there)
            "b": "differs",        # EPP: queues by priority band; lab: warns, keeps shedding priority < 0 with 429
            "c": "differs",        # lab: exact rotation a, b, c, ...; EPP: uniform random tie-breaks
            "d": "lab-rejects",    # the lab router runs exactly one profile
            "e": "lab-rejects",    # the lab's affinity filter supports prefillThroughput only
            "f": "differs"}        # EPP reads the first series (rank 0); the lab sums the ranks
### END SOLUTION

# %% check
import warnings
from igwlab.promtext import Families
from igwlab.router import ConfigError, extract_vllm, load_config
from igwlab.router.plugins import MaxScorePicker

assert verdicts == {"a": "same", "b": "differs", "c": "differs", "d": "lab-rejects", "e": "lab-rejects",
                    "f": "differs"}, verdicts
# the evidence, from the lab's own code:
assert [(s.TYPE, w) for s, w in load_config(epp_config).profile.scorers] == \
       [(s.TYPE, w) for s, w in load_config("default-weighted").profile.scorers]                               # a
fc = load_config({**epp_config, "featureGates": ["flowControl"]})
assert any("NOT implemented" in w for w in fc.warnings)                                                        # b
picker, eps = MaxScorePicker(), [type("E", (), {"name": n})() for n in "abc"]
assert [picker.pick(None, [(e, 0.0) for e in eps])[0].name for _ in range(6)] == list("abcabc")               # c
for bad in ({"apiVersion": "llm-d.ai/v1", "kind": "EndpointPickerConfig", "plugins": [{"type": "queue-scorer"}],
             "schedulingProfiles": [{"name": "prefill", "plugins": [{"pluginRef": "queue-scorer"}]},
                                    {"name": "decode", "plugins": [{"pluginRef": "queue-scorer"}]}]},       # d
            {"apiVersion": "llm-d.ai/v1", "kind": "EndpointPickerConfig",
             "plugins": [{"type": "prefix-cache-affinity-filter", "parameters": {"ttftSource": "latencyPredictor"}}]}):  # e
    try:
        load_config(bad)
        raise AssertionError("expected ConfigError")
    except ConfigError:
        pass
dp = Families.from_text('vllm:num_requests_waiting{engine="0"} 1\nvllm:num_requests_waiting{engine="1"} 6\n')
assert (extract_vllm(dp).waiting, extract_vllm(dp, aggregate="first").waiting) == (7, 1)                        # f
print("✅ a: same; b, c, f: the lab differs; d, e: the lab refuses — measure on kind before trusting a lab result there")

# %% [markdown]
# ## Exercise 4.3 — what an InferencePool selects
#
# An `InferencePool` (v1) names its members with `selector.matchLabels`. Every label must match, and
# the pods must be in the same namespace as the pool. The pool lists up to 8 `targetPorts`. For the EPP,
# **each ready pod × each port is one endpoint** (`podIP:port`).
#
# Write `pool_endpoints(pool_spec, pods)`. Each pod is a dict
# `{"labels": {...}, "ip": "...", "ready": bool}`. Return a sorted list of `"ip:port"` strings.

# %% exercise
def pool_endpoints(pool_spec: dict, pods: list) -> list:
    ### BEGIN SOLUTION
    want = pool_spec["selector"]["matchLabels"]
    ports = [p["number"] for p in pool_spec["targetPorts"]]
    out = []
    for pod in pods:
        if pod["ready"] and all(pod["labels"].get(k) == v for k, v in want.items()):
            out += [f"{pod['ip']}:{port}" for port in ports]
    return sorted(out)
    ### END SOLUTION

# %% check
spec = {"selector": {"matchLabels": {"app": "vllm-sim"}}, "targetPorts": [{"number": 8000}, {"number": 8001}]}
pods = [{"labels": {"app": "vllm-sim", "llm-d.ai/engine-type": "vllm"}, "ip": "10.0.0.5", "ready": True},
        {"labels": {"app": "vllm-sim"}, "ip": "10.0.0.6", "ready": False},
        {"labels": {"app": "other"}, "ip": "10.0.0.7", "ready": True}]
assert pool_endpoints(spec, pods) == ["10.0.0.5:8000", "10.0.0.5:8001"]
assert pool_endpoints({"selector": {"matchLabels": {"app": "vllm-sim", "tier": "a"}}, "targetPorts": [{"number": 8000}]}, pods) == []
print("✅ one endpoint per ready matching pod per target port (e.g. one vLLM data-parallel rank per port)")

# %% [markdown]
# Exercises 4.2 f and 4.3 fit together. Take a data-parallel pod that serves all ranks behind **one**
# port. It is one endpoint, and its metrics have several `engine` series. The EPP reads only the
# first series. Another pod serves each rank on **its own** port, and `targetPorts` lists these ports.
# That pod gives the EPP one endpoint per rank.
#
# ## Exercise 4.4 — read a data-parallel pod like the EPP does
#
# The next cell holds **sample output in the documented vLLM/llm-d-inference-sim format (illustrative)**
# for a pod with two data-parallel ranks. It does not come from a live pod. Write `epp_view(text)`,
# which returns `(waiting, running, kv_usage, running_loras)`. Read the text as the
# `core-metrics-extractor` (v0.10.0) of the EPP reads it:
#
# - For each gauge, it keeps **one** series: the series with the latest timestamp. The vLLM output
#   has no timestamp, so this is the first series in the list.
# - It adds no series together.
# - The LoRA adapters come from the `running_lora_adapters` label of the `vllm:lora_requests_info`
#   series with the **largest value**. The value is a timestamp.

# %% exercise
SAMPLE = """# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="lab/llm"} 5.0
vllm:num_requests_running{engine="1",model_name="lab/llm"} 8.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="lab/llm"} 2.0
vllm:num_requests_waiting{engine="1",model_name="lab/llm"} 9.0
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="lab/llm"} 0.4375
vllm:kv_cache_usage_perc{engine="1",model_name="lab/llm"} 0.9125
# HELP vllm:lora_requests_info Running stats on lora requests.
# TYPE vllm:lora_requests_info gauge
vllm:lora_requests_info{max_lora="2",running_lora_adapters="sql-lora",waiting_lora_adapters=""} 1.7907e+09
vllm:lora_requests_info{max_lora="2",running_lora_adapters="sql-lora,chat-lora",waiting_lora_adapters=""} 1.7907001e+09
# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",engine="0",num_gpu_blocks="2048"} 1.0
"""

def epp_view(text: str):
    ### BEGIN SOLUTION
    fam = Families.from_text(text)
    first = lambda name: fam.get(name)[0].value
    lora = max(fam.get("vllm:lora_requests_info"), key=lambda s: s.value)
    loras = {x for x in (lora.label("running_lora_adapters") or "").split(",") if x}
    return int(first("vllm:num_requests_waiting")), int(first("vllm:num_requests_running")), \
        first("vllm:kv_cache_usage_perc"), loras
    ### END SOLUTION

# %% check
w, r, kv, loras = epp_view(SAMPLE)
m = extract_vllm(Families.from_text(SAMPLE), aggregate="first")
assert (w, r, kv, loras) == (m.waiting, m.running, m.kv_usage, m.active_models) == (2, 5, 0.4375, {"sql-lora", "chat-lora"})
lab = extract_vllm(Families.from_text(SAMPLE))
print(f"EPP view: waiting {w}, KV {kv} | lab router view (sum ranks, max KV): waiting {lab.waiting}, KV {lab.kv_usage}")
print("✅ to the EPP this pod looks far less loaded than it is: rank 1 (9 waiting, KV 0.91) is invisible")

# %% [markdown]
# ## Measure whatever stack is running (T0, or T0 + Docker)
#
# The next cell looks for a router in this sequence:
#
# 1. `IGW_ROUTER_URL`, if you set it.
# 2. The kind port-forward (`kubectl port-forward svc/igw-epp 8081:8081`, which gives
#    `http://localhost:8081`).
# 3. The compose router (`http://localhost:9000`).
#
# If it finds none, it starts the in-process stack. Then it runs the same agentic sessions and tells
# what it measured.
#
# The hit rate comes from `usage.prompt_tokens_details.cached_tokens`. Not all engines send this
# field:
#
# - The fake backend always sends it.
# - `llm-d-inference-sim` v0.11.2 sends it when `--enable-kvcache` is on (the stacks of the lab set
#   it). The simulator counts the value in its own tokens and 16-token blocks.
# - **vLLM sends it only when started with `--enable-prompt-tokens-details`** (the GPU and GKE paths
#   of the lab set it).
#
# For an engine that does not send it, the result shows `n/a`, never 0%. For such an engine, the
# vLLM counters `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` give the hit rate
# instead (`igwlab.bench.engine_hit_rate`, the method of Exercise 1.4).

# %%
from igwlab.bench import compare, engine_hit_rate, run_bench


def reachable(url):
    try:
        with urllib.request.urlopen(url + "/v1/models", timeout=1) as r:
            return r.status == 200
    except Exception:
        return False


sessions = agentic_sessions(n_sessions=9, turns=3, n_agents=3, system_words=800, tool_words=200, seed=2)
candidates = [(os.environ["IGW_ROUTER_URL"], "the router at IGW_ROUTER_URL (real engines if that is what runs behind it)")] \
    if os.environ.get("IGW_ROUTER_URL") else []
candidates += [("http://localhost:8081", "the kind stack: the llm-d EPP in front of llm-d-inference-sim (emulated timing)"),
               ("http://localhost:9000", "the compose stack: the lab router in front of the simulator or the fake (emulated timing)")]
target = next(((u, what) for u, what in candidates if reachable(u)), None)
if target:
    url, what = target
    with urllib.request.urlopen(url + "/v1/models", timeout=5) as r:
        models = [m["id"] for m in json.load(r)["data"]]
    print(f"benchmarking {what} at {url} (model {models[0]})")
    result = run_bench(url, sessions, model=models[0], label="running stack")
else:
    print("no running stack reachable (fine on Colab/CI): benchmarking the in-process stack (T0, emulated timing)")
    with LocalStack(3, "default-weighted") as s:
        result = s.bench(sessions, label="in-process")
print(compare([result]))

# %% [markdown]
# ## T1/T2: the lab router in front of real vLLM
#
# If you have a GPU, start two or more vLLM replicas with `deploy/any-gpu/serve.sh` (pip) or its
# `docker-compose.yaml`. Then set `IGW_BACKENDS=r0=http://127.0.0.1:8001,r1=http://127.0.0.1:8002`
# (the value that `serve.sh` prints). For each policy, the next cell puts a new in-process lab router
# in front of those replicas. It emulates nothing about the engines. It prints:
#
# 1. round-robin against the llm-d default weights on the same agent sessions. For each, it gives
#    TTFT, the hit rate and the per-replica split. The hit rate comes from the responses, and also
#    from the counters of vLLM.
# 2. the live `vllm:num_requests_waiting/running` from each replica, and the proposal of the HPA of
#    notebook 03. The targets come from two values of *these* replicas. One value is `IGW_SLOTS`
#    batch slots (the `--max-num-seqs` of serve.sh, default 16). The other value is the measured
#    time that a request holds its slot.
#
# If you do not set `IGW_BACKENDS`, the cell prints the commands and does nothing else (T0 stays offline).

# %%
from igwlab.autoscale import recommend_from_scrapes, waiting_target


def server_kind(url: str) -> str:
    """What answers at `url`: the lab's fake, llm-d-inference-sim, or a real engine."""
    with urllib.request.urlopen(url + "/v1/models", timeout=5) as r:
        models = json.load(r)["data"]
    if models and models[0].get("owned_by") == "igwlab":
        return "the lab's fake backend (emulated timing)"
    body = json.dumps({"model": models[0]["id"], "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        simulated = "x-inference-pod" in {k.lower() for k in r.headers}      # llm-d-inference-sim names itself
    return "llm-d-inference-sim (emulated timing)" if simulated else "a real engine (measured)"


spec = os.environ.get("IGW_BACKENDS", "")
backends = dict(item.split("=", 1) for item in spec.split(",") if "=" in item)
live = {n: u for n, u in backends.items() if reachable(u.rstrip("/"))}
if not backends:
    print("IGW_BACKENDS is not set: no GPU replicas to measure. On a GPU machine (Colab/Kaggle T4, a rented GPU):")
    print(f"  pip install 'vllm==0.30.0'   # (verify)\n  REPLICAS=2 bash {DEPLOY / 'any-gpu/serve.sh'}")
    print("  export IGW_BACKENDS=r0=http://127.0.0.1:8001,r1=http://127.0.0.1:8002   # then re-run this cell")
elif live != backends:
    print("not reachable:", sorted(set(backends) - set(live)), "— start them with deploy/any-gpu/serve.sh")
else:
    model = os.environ.get("IGW_MODEL", "lab/llm")                  # serve.sh serves the model as lab/llm
    slots = int(os.environ.get("IGW_SLOTS", "16"))
    measured, counter_rates = [], {}
    for seed, cfg in ((3, "round-robin"), (13, "default-weighted")):
        # These servers outlive each run, so their caches do too: each policy gets its own agent
        # programs (another seed, same shape) so the second run cannot reuse the first run's prefixes.
        gpu_sessions = agentic_sessions(n_sessions=12, turns=4, n_agents=3, system_words=800, tool_words=250, seed=seed)
        with LocalStack(config=cfg, backends=live, model=model) as s:
            before = s.backend_metrics()
            measured.append(s.bench(gpu_sessions, label=cfg))
            counter_rates[cfg] = engine_hit_rate(before, s.backend_metrics())
    kinds = {n: server_kind(u.rstrip("/")) for n, u in live.items()}
    print("servers:", kinds)
    print(compare(measured))
    print("hit rate from vLLM's own counters:", {k: None if v is None else f"{v:.1%}" for k, v in counter_rates.items()})

    with LocalStack(config="default-weighted", backends=live, model=model) as s:
        import threading
        load = threading.Thread(target=lambda: s.bench(agentic_sessions(n_sessions=48, turns=3, n_agents=3, seed=23),
                                                       label="load", stagger_s=0.2))
        load.start()
        scrapes = []
        while load.is_alive():
            scrapes.append(s.backend_metrics())
            load.join(timeout=1.0)
        last = Families.from_text("".join(s.backend_metrics().values()))
    n_e2e = last.sum("vllm:e2e_request_latency_seconds_count") or 1
    n_q = last.sum("vllm:request_queue_time_seconds_count") or 1
    service_s = last.sum("vllm:e2e_request_latency_seconds_sum", 0.0) / n_e2e - \
        last.sum("vllm:request_queue_time_seconds_sum", 0.0) / n_q
    targets = {"vllm:num_requests_waiting": waiting_target(0.5, max(service_s, 1e-3), slots),
               "vllm:num_requests_running": 0.75 * slots}
    print(f"\nmeasured: a request holds its slot for {service_s:.2f} s -> targets {targets} "
          f"({slots} slots, 0.5 s queueing budget)")
    for sc in scrapes[:: max(1, len(scrapes) // 8)]:
        best, per = recommend_from_scrapes(sc, targets, len(live))
        print({m.split(":")[1]: per[m][0] for m in targets}, f"-> proposal {best} for {len(live)} replicas "
              "(before the min/max clamp and the behavior policies)")

# %% [markdown]
# ## In a design review
#
# **Two-minute walkthrough.** "Locally, we run the real control plane against fake GPUs. Three
# `llm-d-inference-sim` pods speak the OpenAI API and export the metric names of vLLM. An
# `InferencePool` selects them by label. The llm-d Router runs in standalone mode: Envoy is in front,
# and it asks the EPP over ext-proc for each request. The plugin config of the EPP is the same
# document that we adjusted in-process.
#
# "The simulator reproduces the TTFT formula and the prefix cache of each request, but not prefill
# contention. Thus, on the simulator, we compare hit rates and per-replica splits, not TTFT. We also
# know where the lab router and the EPP are different: how they break exact ties, flow control and
# data-parallel metrics.
#
# "On one rented GPU, we then put the same router in front of real vLLM replicas. We calculate the
# autoscaling targets again from measured numbers. In production, two things change: the model
# servers (real vLLM on GPUs) and the proxy (a cloud load balancer in Gateway mode). No local stack
# tells us these things: cold-start times, network effects and the traffic of the real fleet."
#
# **Drill questions**
#
# 1. *What is the difference between standalone mode and Gateway mode?* In standalone mode, a
#    self-managed Envoy (a sidecar or its own Deployment) is in front of the EPP. You do not need the
#    Gateway API. In Gateway mode, an `HTTPRoute` on a shared Gateway points at the `InferencePool`.
#    Then the gateway implementation (Envoy Gateway, Istio, agentgateway, the regional load balancer
#    of GKE) calls the EPP.
# 2. *Why must you install the CRDs before the Helm chart?* The chart creates an `InferencePool` and
#    `InferenceObjective` objects. Without their CRDs, the API server rejects them, and the install
#    fails.
# 3. *The bench shows a 0% hit rate against vLLM. Is prefix caching off?* Probably not. vLLM does not
#    send `prompt_tokens_details` unless you start it with `--enable-prompt-tokens-details`. A bench
#    that reads an absent field as 0 gives a false result. To find the real hit rate, read
#    `vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total` (their increases during the
#    run).
