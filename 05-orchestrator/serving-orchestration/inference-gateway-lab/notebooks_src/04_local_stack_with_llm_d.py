# %% [markdown]
# # 04 · The local stack with llm-d
#
# **Tier:** T0 walkthrough on any machine (reads the deploy files, dry-runs the scripts, parses
# sample metrics); **T1-local** when Docker + kind are available (still CPU only — the model servers
# are simulators). If a stack is running, the notebook benchmarks it; otherwise it benchmarks the
# in-process stack instead and says so.
#
# ## The one-minute version
#
# The decision logic you exercised in notebooks 01–02 exists as production components:
#
# | piece | in-process (T0) | docker compose (`deploy/local`) | kind (`deploy/kind`) |
# |---|---|---|---|
# | model servers | `igwlab.fakebackend` | 3 × `llm-d-inference-sim` (or the fake) | 3 × `llm-d-inference-sim` pods |
# | proxy | the lab router | the lab router | Envoy (sidecar), ext-proc to the EPP |
# | endpoint picker | the lab router | the lab router | **llm-d EPP** (Helm, standalone chart) |
# | pool definition | a Python list | router flags | `InferencePool` v1 (selector + target ports) |
# | request classes | `RouterSettings.objectives` | `--objective` flags | `InferenceObjective` CRs |
# | metrics | `/metrics` | Prometheus | `/metrics` (+ ServiceMonitor if enabled) |
#
# The EndpointPickerConfig is the same document in all three (the kind Helm values embed the lab's
# `default-weighted` preset verbatim), and the simulator is configured with the same latency model
# as the fake backend. Background: [PRIMER §9 The Kubernetes-native stack, Sep 2026 and §10 Where to
# run it](../../PRIMER.md).

# %%
import shutil
import subprocess
from pathlib import Path

import yaml

LAB = Path.cwd().resolve()
while not (LAB / "igwlab").exists():
    LAB = LAB.parent
DEPLOY = LAB / "deploy"
tools = {t: bool(shutil.which(t)) for t in ("docker", "kind", "kubectl", "helm")}
print("tools on this machine:", tools)

# %% [markdown]
# ## Path 1: docker compose
#
# `deploy/local/docker-compose.yaml` has two profiles: `sim` (the upstream simulator image) and `fake`
# (the bundled fake backend, no registry pull). The router is the lab router with the
# `default-weighted` preset; Prometheus scrapes everything.

# %%
compose = yaml.safe_load((DEPLOY / "local/docker-compose.yaml").read_text())
for name, svc in compose["services"].items():
    print(f"{name:<12} profiles={svc.get('profiles')} image={svc.get('image')}")
print("\nsimulator flags:", " ".join(compose["services"]["sim-a"]["command"]))

# %% [markdown]
# The simulator's `per-token` latency calculator is the fake backend's model:
# `TTFT = prefill-overhead + (prompt − cached) × prefill-time-per-token`, and `inter-token-latency`
# per output token. `EngineProfile.sim_args()` prints the flags for a profile.
#
# ## Exercise 4.1 — read the simulator's latency model from its flags
#
# Parse the flag list (either `--flag value` or `--flag=value`; Go durations like `2ms`, `50us`,
# `1.5s`) and return the TTFT in seconds the simulator emulates for a prompt of `prompt_tokens` of
# which `cached_tokens` hit the prefix cache (queueing ignored).

# %% exercise
import re

def sim_ttft(args, prompt_tokens: int, cached_tokens: int) -> float:
    ### BEGIN SOLUTION
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

    def dur(s):
        n, unit = re.fullmatch(r"([0-9.]+)(us|µs|ms|s)", s).groups()
        return float(n) * {"us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.0}[unit]
    return dur(flags["prefill-overhead"]) + (prompt_tokens - cached_tokens) * dur(flags["prefill-time-per-token"])
    ### END SOLUTION

# %% check
from igwlab.fakebackend import EngineProfile
p = EngineProfile()
kind_args = yaml.safe_load((DEPLOY / "kind/sim-deployment.yaml").read_text())["spec"]["template"]["spec"]["containers"][0]["args"]
for args in (compose["services"]["sim-a"]["command"], kind_args, p.sim_args()):
    for n, c in ((2000, 0), (2000, 1600), (64, 0)):
        assert abs(sim_ttft(args, n, c) - p.prefill_seconds(n, c)) < 1e-12
print(f"✅ 2,000-token prompt: {sim_ttft(kind_args, 2000, 0) * 1e3:.0f} ms cold, "
      f"{sim_ttft(kind_args, 2000, 1600) * 1e3:.0f} ms with 1,600 cached — same model in all three paths")

# %% [markdown]
# ## Path 2: kind + the real llm-d Router (standalone mode)
#
# `deploy/kind/up.sh` creates a cluster, installs the two CRDs (InferencePool v1 from the Gateway API
# Inference Extension, InferenceObjective from llm-d-router), deploys three simulator pods, and installs
# the `llm-d-router-standalone` Helm chart: an EPP Deployment with an Envoy sidecar that receives
# client traffic on :8081 and asks the EPP (ext-proc) where to send each request. `DRY_RUN=1` prints
# the steps without running anything:

# %%
out = subprocess.run(["bash", str(DEPLOY / "kind/up.sh")], env={"DRY_RUN": "1", "PATH": "/usr/bin:/bin"},
                     capture_output=True, text=True, check=True).stdout
print("\n".join(l for l in out.splitlines() if l.startswith(("==>", "+ "))))

# %% [markdown]
# ## Exercise 4.2 — the config the EPP will run
#
# The chart mounts `router.epp.pluginsCustomConfig[<router.epp.pluginsConfigFile>]` as the EPP's
# `--config-file`. Write `epp_config(values)` that returns that document as a dict, then confirm with
# the lab's loader that the EPP on kind runs exactly the policy you measured in notebooks 01–02.

# %% exercise
def epp_config(values: dict) -> dict:
    ### BEGIN SOLUTION
    epp = values["router"]["epp"]
    return yaml.safe_load(epp["pluginsCustomConfig"][epp["pluginsConfigFile"]])
    ### END SOLUTION

# %% check
from igwlab.router import load_config
values = yaml.safe_load((DEPLOY / "kind/router-values.yaml").read_text())
cfg = load_config(epp_config(values))
preset = load_config("default-weighted")
assert [(s.TYPE, w) for s, w in cfg.profile.scorers] == [(s.TYPE, w) for s, w in preset.profile.scorers]
print(cfg.describe())
print("objectives created by the chart:", values["router"]["inferenceObjectives"])
print("✅ same scorers and weights as the in-process router")

# %% [markdown]
# ## Exercise 4.3 — what an InferencePool selects
#
# An `InferencePool` (v1) names its members with `selector.matchLabels` (every label must match, same
# namespace) and lists up to 8 `targetPorts`; **each ready pod × each port is one endpoint**
# (`podIP:port`) for the EPP. Write `pool_endpoints(pool_spec, pods)` for pods given as dicts
# `{"labels": {...}, "ip": "...", "ready": bool}`; return a sorted list of `"ip:port"` strings.

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
# ## Exercise 4.4 — read a simulator's metrics like the EPP does
#
# Below is **sample output in the documented vLLM/llm-d-inference-sim format (illustrative)** — not
# captured from a live pod. Write `routing_view(text)` → `(waiting, running, kv_usage, running_loras)`
# the way the EPP's `core-metrics-extractor` reads it: sum the gauges over engines, KV usage is the
# 0–1 gauge, and LoRA adapters come from the `running_lora_adapters` label of the
# `vllm:lora_requests_info` series with the **latest** value (the value is a timestamp).

# %% exercise
SAMPLE = """# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="lab/llm"} 5.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="lab/llm"} 2.0
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="lab/llm"} 0.4375
# HELP vllm:lora_requests_info Running stats on lora requests.
# TYPE vllm:lora_requests_info gauge
vllm:lora_requests_info{max_lora="2",running_lora_adapters="sql-lora",waiting_lora_adapters=""} 1.7907e+09
vllm:lora_requests_info{max_lora="2",running_lora_adapters="sql-lora,chat-lora",waiting_lora_adapters=""} 1.7907001e+09
# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",engine="0",num_gpu_blocks="2048"} 1.0
"""
from igwlab.promtext import Families

def routing_view(text: str):
    ### BEGIN SOLUTION
    f = Families.from_text(text)
    lora = max(f.get("vllm:lora_requests_info"), key=lambda s: s.value)
    loras = {x for x in (lora.label("running_lora_adapters") or "").split(",") if x}
    return (int(f.sum("vllm:num_requests_waiting", 0)), int(f.sum("vllm:num_requests_running", 0)),
            f.max("vllm:kv_cache_usage_perc", 0.0), loras)
    ### END SOLUTION

# %% check
from igwlab.router import extract_vllm
w, r, kv, loras = routing_view(SAMPLE)
m = extract_vllm(Families.from_text(SAMPLE))
assert (w, r, kv, loras) == (m.waiting, m.running, m.kv_usage, m.active_models) == (2, 5, 0.4375, {"sql-lora", "chat-lora"})
print("✅", m)

# %% [markdown]
# ## Measure whatever stack is running
#
# The next cell looks for a router in this order: the kind port-forward (`kubectl port-forward
# svc/igw-epp 8081:8081` → `http://localhost:8081`), the compose router (`http://localhost:9000`), and
# otherwise starts the in-process stack. It then runs the same agentic sessions and prints where the
# numbers came from. (Against the simulator the `cached_tokens` usage field may be absent, so the hit
# rate can read 0% there even though the simulator's own `vllm:prefix_cache_hits_total` counts hits.)

# %%
import urllib.request

from igwlab.bench import agentic_sessions, compare, run_bench
from igwlab.stack import LocalStack


def reachable(url):
    try:
        with urllib.request.urlopen(url + "/v1/models", timeout=1) as r:
            return r.status == 200
    except Exception:
        return False


sessions = agentic_sessions(n_sessions=9, turns=3, n_agents=3, system_words=800, tool_words=200, seed=2)
target = next((u for u in ("http://localhost:8081", "http://localhost:9000") if reachable(u)), None)
if target:
    models = []
    with urllib.request.urlopen(target + "/v1/models", timeout=5) as r:
        import json
        models = [m["id"] for m in json.load(r)["data"]]
    print(f"benchmarking the running stack at {target} (model {models[0]}) — timing is whatever that stack emulates")
    result = run_bench(target, sessions, model=models[0], label="running stack")
else:
    print("no local stack reachable (fine on Colab/CI): benchmarking the in-process stack instead (T0, emulated timing)")
    with LocalStack(3, "default-weighted") as s:
        result = s.bench(sessions, label="in-process")
print(compare([result]))

# %% [markdown]
# ## In a design review
#
# **Two-minute walkthrough.** "Locally we run the real control plane against fake GPUs: three
# `llm-d-inference-sim` pods that speak the OpenAI API and export vLLM's metric names, an
# `InferencePool` selecting them by label, and the llm-d Router in standalone mode — Envoy in front,
# asking the EPP over ext-proc for every request. The EPP's plugin config is the same document we
# tuned in-process, and the simulator runs the same latency model, so behaviour carries over: what
# changes in production is the model servers (real vLLM on GPUs) and the proxy (a cloud load balancer
# in Gateway mode). What the local stack cannot tell you: real prefill/decode speeds, GPU memory
# limits, cold-start times and network effects."
#
# **Drill questions**
#
# 1. *Standalone vs Gateway mode?* Standalone: a self-managed Envoy (sidecar or its own Deployment)
#    in front of the EPP, no Gateway API needed. Gateway mode: an `HTTPRoute` on a shared Gateway
#    points at the `InferencePool`, and the gateway implementation (Envoy Gateway, Istio,
#    agentgateway, GKE's regional load balancer) calls the EPP.
# 2. *Why must the CRDs be installed before the Helm chart?* The chart creates an `InferencePool` and
#    `InferenceObjective` objects; without their CRDs the API server rejects them and the install fails.
# 3. *What does `failureMode: FailOpen` on the InferencePool mean?* If the EPP is unreachable the proxy
#    still forwards the request (to some pool member) instead of failing it (`FailClose`).
