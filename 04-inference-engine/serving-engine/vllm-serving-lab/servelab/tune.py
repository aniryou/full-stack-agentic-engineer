"""tune.py — sweep engine flags against an SLO, on any backend.

One idea: every engine flag is a trade-off, so the only honest way to choose one is to measure
the **goodput** each setting achieves on *your* workload at *your* SLO. A sweep is a loop:
(re)start the engine with a config, warm up, replay the same seeded workload, summarize, stop.
The backend decides what "start the engine" means — the fake server (T0, seconds per config), a
local ``vllm serve`` subprocess (T1, a minute or two per config), or a URL you redeploy yourself
(T3). The rest of the loop is identical, which is the point.

    trials = sweep(FakeBackend(), grid(max_num_seqs=[8, 32, 128]), lambda: random_requests(80), rate=8,
                   slo=SLO(ttft_ms=300, tpot_ms=40))
    print(trials_table(trials)); print(best(trials).config)
"""
from __future__ import annotations

import itertools
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from . import env
from . import metrics as M
from .bench.runner import BenchRun, run_closed_loop, run_open_loop, warm_up
from .bench.summary import SLO, Summary
from .fake_engine import EngineConfig, profile as named_profile


def to_cli_flags(config: dict) -> list:
    """``{"max_num_seqs": 64, "enable_prefix_caching": False, "speculative_config": {...}}`` ->
    ``["--max-num-seqs", "64", "--no-enable-prefix-caching", "--speculative-config", "{...}"]``.
    Booleans use vLLM's ``--flag`` / ``--no-flag`` pairs; dicts become JSON."""
    out = []
    for key, value in config.items():
        flag = "--" + key.replace("_", "-")
        if value is None:
            continue
        if value is True:
            out.append(flag)
        elif value is False:
            out.append("--no-" + key.replace("_", "-"))
        elif isinstance(value, (dict, list)):
            out += [flag, json.dumps(value, separators=(",", ":"))]
        else:
            out += [flag, str(value)]
    return out


def grid(**axes) -> list:
    """Cartesian product of flag values: ``grid(a=[1, 2], b=[True])`` -> ``[{a:1,b:True}, {a:2,b:True}]``."""
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(axes[k] for k in keys))]


# ---------------------------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------------------------
_PROFILE_KNOBS = {"gpu_memory_utilization", "max_model_len", "block_size", "kv_cache_dtype", "quantization", "dtype",
                  "num_gpu_blocks_override"}
_ENGINE_KNOBS = {"max_num_seqs", "max_num_batched_tokens", "enable_prefix_caching", "enable_chunked_prefill",
                 "long_prefill_token_threshold"}
# Draft time per proposed token, as a fraction of the target's weight-read time (assumptions).
DRAFT_COST = {"ngram": 0.0, "suffix": 0.0, "eagle": 0.05, "eagle3": 0.05, "mtp": 0.05, "draft_model": 0.15}
# ``speculative_config`` keys the fake backend reads or tolerates: all are real vLLM
# ``SpeculativeConfig`` fields (v0.30.0), so a config that runs here also runs on ``vllm serve``.
_SPEC_KEYS = {"method", "model", "num_speculative_tokens", "prompt_lookup_max", "prompt_lookup_min",
              "draft_tensor_parallel_size", "quantization", "max_model_len", "revision"}


class FakeBackend:
    """The fake server with vLLM-named knobs. Speculation needs an acceptance rate, which in real
    life is a property of the workload and the draft method — here it is an explicit assumption
    passed as ``FakeBackend(spec_acceptance=...)``, never smuggled into ``speculative_config``
    (vLLM would reject an unknown key there)."""
    simulated = True

    def __init__(self, profile: str = "t4-qwen2.5-0.5b", spec_acceptance: float = 0.6, time_scale: float = 1.0,
                 **profile_overrides):
        self.profile_name, self.spec_acceptance, self.time_scale = profile, spec_acceptance, time_scale
        self.profile_overrides = profile_overrides
        self.server = None

    def build(self, config: dict):
        unknown = set(config) - _PROFILE_KNOBS - _ENGINE_KNOBS - {"speculative_config"}
        if unknown:
            raise ValueError(f"the fake backend does not model {sorted(unknown)}")
        knobs = {k: v for k, v in config.items() if k in _PROFILE_KNOBS}
        if "num_gpu_blocks_override" in knobs:          # vLLM's flag for testing preemption
            knobs["num_blocks"] = knobs.pop("num_gpu_blocks_override")
        prof = named_profile(self.profile_name, **{**self.profile_overrides, **knobs})
        ecfg = EngineConfig(**{k: v for k, v in config.items() if k in _ENGINE_KNOBS})
        spec = config.get("speculative_config")
        if spec and set(spec) - _SPEC_KEYS:
            raise ValueError(f"speculative_config keys {sorted(set(spec) - _SPEC_KEYS)} are not vLLM "
                             "SpeculativeConfig fields this backend models; pass the acceptance assumption "
                             "as FakeBackend(spec_acceptance=...)")
        if spec:
            ecfg.num_speculative_tokens = int(spec.get("num_speculative_tokens", 0))
            ecfg.spec_acceptance = float(self.spec_acceptance)
            ecfg.spec_draft_cost = DRAFT_COST.get(spec.get("method", "ngram"), 0.1)
        return prof, ecfg

    def start(self, config: dict) -> str:
        from .fakeserver import FakeServer
        prof, ecfg = self.build(config)
        self.server = FakeServer(prof, ecfg, time_scale=self.time_scale)
        return self.server.start()

    def stop(self) -> None:
        if self.server:
            self.server.stop()
            self.server = None


class VLLMBackend:
    """``vllm serve <model> <flags>`` as a subprocess on this machine (T1: needs a GPU + vLLM).
    Each config costs a model load and a profile run; keep grids small."""
    simulated = False

    def __init__(self, model: str, base_flags: dict | None = None, port: int = 8000, host: str = "127.0.0.1",
                 startup_timeout_s: float = 900, log_dir: str = ".", executable: str = "vllm"):
        self.model, self.base_flags, self.port, self.host = model, dict(base_flags or {}), port, host
        self.startup_timeout_s, self.log_dir, self.executable = startup_timeout_s, Path(log_dir), executable
        self.proc: subprocess.Popen | None = None
        self.log_path: Path | None = None

    @staticmethod
    def available() -> bool:
        return env.has_gpu() and env.has_vllm()

    def command(self, config: dict) -> list:
        return [self.executable, "serve", self.model, "--host", self.host, "--port", str(self.port),
                *to_cli_flags({**self.base_flags, **config})]

    def start(self, config: dict) -> str:
        cmd = self.command(config)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"vllm-{int(time.time())}.log"
        print("$", " ".join(cmd), f"  (log: {self.log_path})")
        log = open(self.log_path, "w")  # noqa: SIM115 — closed with the process
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ})
        url = f"http://{self.host}:{self.port}"
        deadline = time.time() + self.startup_timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vllm exited with {self.proc.returncode}:\n{self.startup_log()[-3000:]}")
            if env.wait_healthy(url, timeout_s=5):
                return url
        self.stop()
        raise TimeoutError(f"vllm not healthy after {self.startup_timeout_s}s; see {self.log_path}")

    def startup_log(self) -> str:
        return self.log_path.read_text() if self.log_path and self.log_path.exists() else ""

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


class URLBackend:
    """An engine someone else starts (Cloud Run, GKE): flags change by redeploying, so each config
    must already be live when ``start`` is called — this backend only points the bench at it."""
    simulated = False

    def __init__(self, url: str):
        self.url = url

    def start(self, config: dict) -> str:
        if config:
            print(f"URLBackend: make sure {self.url} runs with {to_cli_flags(config)} (redeploy to change flags)")
        return self.url

    def stop(self) -> None:
        pass


# ---------------------------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------------------------
@dataclass
class Trial:
    config: dict
    summary: Summary
    snapshot: M.EngineSnapshot | None
    run: BenchRun

    def row(self, percentile_: int = 99) -> dict:
        r = {"config": ", ".join(f"{k}={v}" for k, v in self.config.items()) or "(defaults)"}
        r.update({k: v for k, v in self.summary.row(percentile_).items() if k != "label"})
        if self.snapshot is not None:
            r["preempt"] = self.snapshot.preemptions
            r["hit rate"] = self.snapshot.prefix_hit_rate
        return r


def sweep(backend, configs: list, workload, *, rate: float | None = None, concurrency: int | None = None,
          slo: SLO | None = None, warmup: int = 2, burstiness: float = 1.0, seed: int = 0,
          headers: dict | None = None, scrape_metrics: bool = True) -> list:
    """Run the same workload against each config. ``workload`` is a list of requests or a
    zero-argument function returning one (called per config so every trial gets fresh objects)."""
    if (rate is None) == (concurrency is None):
        raise ValueError("give exactly one of rate= (open loop) or concurrency= (closed loop)")
    trials = []
    for cfg in configs:
        url = backend.start(cfg)
        try:
            reqs = workload() if callable(workload) else list(workload)
            if warmup:
                warm_up(url, reqs, warmup, headers=headers)
            before = M.scrape(url, headers=headers) if scrape_metrics else None
            if rate is not None:
                run = run_open_loop(url, reqs, rate, burstiness=burstiness, seed=seed, headers=headers)
            else:
                run = run_closed_loop(url, reqs, concurrency, headers=headers)
            snap = M.snapshot(M.scrape(url, headers=headers), before) if scrape_metrics else None
        finally:
            backend.stop()
        label = ", ".join(f"{k}={v}" for k, v in cfg.items())
        trials.append(Trial(cfg, run.summary(slo, label=label), snap, run))
    return trials


def trials_table(trials: list, percentile_: int = 99) -> str:
    from .bench.report import table
    return table([t.row(percentile_) for t in trials])


def best(trials: list, min_attainment: float = 0.9, objective: str = "output_throughput"):
    """The trial with the highest ``objective`` among those meeting the SLO for at least
    ``min_attainment`` of requests; ``None`` when no config meets it."""
    feasible = [t for t in trials if t.summary.slo_attainment >= min_attainment]
    return max(feasible, key=lambda t: getattr(t.summary, objective)) if feasible else None


def max_rate_under_slo(measure, lo: float, hi: float, min_attainment: float = 0.9, iters: int = 6) -> tuple:
    """Highest open-loop rate whose SLO attainment stays >= ``min_attainment``, by bisection in
    log space. ``measure(rate) -> attainment`` runs one benchmark (inject a fake for tests).
    Assumes attainment falls as rate rises. Returns ``(rate, [(rate, attainment), ...])``."""
    history = []
    a_lo = measure(lo)
    history.append((lo, a_lo))
    if a_lo < min_attainment:
        return math.nan, history
    a_hi = measure(hi)
    history.append((hi, a_hi))
    if a_hi >= min_attainment:
        return hi, history
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        a = measure(mid)
        history.append((mid, a))
        if a >= min_attainment:
            lo = mid
        else:
            hi = mid
    return lo, history
