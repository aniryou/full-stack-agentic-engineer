"""The T1 cells of notebooks 01-04: the gateway in front of one real vLLM, as functions the tests can run.

The one idea: a number is *measured* only if the real engine produced it. Every function here starts its own
in-process gateway with the `vllm` config (vLLM first in `chat`, a fake provider second), and labels each result by
the target that actually served it -- `MEASURED` for vLLM (`local/llm`), `SIMULATED (...)` when the chain fell back
to the fake -- so a stopped or unreachable vLLM can never print fake latencies and token counts as measurements.
The notebooks call these only when `env.vllm_url()` found a vLLM that answers `/health`; `tests/test_t1.py` runs
all of them offline against a vLLM stand-in (a fake provider serving `lab/llm`), including a vLLM that dies mid-run.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import uuid

from . import bench, client, promtext
from .fakes import FakeSpec
from .gateway import metering
from .stack import LocalStack

VLLM_TARGET = "local/llm"                    # the catalogue id the `vllm` config gives the real engine
RESTART_HINT = ("restart vLLM before notebooks 03 and 04: `bash deploy/any-gpu/serve.sh` (the fake and the gateway "
                "it started keep running), or `docker compose -f deploy/any-gpu/docker-compose.yaml start vllm`")


def label(result) -> str:
    """MEASURED only when vLLM served this request; otherwise say what did."""
    target = result.header("x-gwlab-target") if hasattr(result, "header") else getattr(result, "target", "")
    if target == VLLM_TARGET:
        return "MEASURED"
    return f"SIMULATED (served by {target or 'nobody'}, not vLLM)"


def vllm_stack(vllm_url: str, fallback: FakeSpec | None = None, overrides: dict | None = None) -> LocalStack:
    """A gateway (config `vllm`) in front of the vLLM at `vllm_url`, with a fake `acme` as the fallback."""
    return LocalStack(config="vllm", upstreams={"local": vllm_url}, fakes={"acme": fallback or FakeSpec(name="acme")},
                      overrides=overrides).start()


# --------------------------------------------------------------------------------------------- notebook 01
def first_requests(vllm_url: str, questions, fallback: FakeSpec | None = None) -> list:
    """Stream each question through the gateway and return (label, target, TTFT s, usage) per request."""
    s = vllm_stack(vllm_url, fallback)
    try:
        k = s.issue_key("team-a")
        out = []
        for q in questions:
            r = s.chat(k, q, stream=True, stream_options={"include_usage": True})
            out.append((label(r), r.header("x-gwlab-target"), r.ttft_s, r.usage))
        return out
    finally:
        s.stop()


# --------------------------------------------------------------------------------------------- notebook 02
def stop_vllm() -> str:
    """Stop the vLLM that deploy/any-gpu started: `vllm serve` (serve.sh), else the compose service. Returns what ran."""
    if shutil.which("pkill") and subprocess.run(["pkill", "-f", "vllm serve"], capture_output=True).returncode == 0:
        return "pkill -f 'vllm serve'"
    compose = os.path.join(os.path.dirname(__file__), "..", "deploy", "any-gpu", "docker-compose.yaml")
    if shutil.which("docker"):
        subprocess.run(["docker", "compose", "-f", compose, "stop", "vllm"], capture_output=True)
        return "docker compose stop vllm"
    return "nothing found to stop"


def stop_midrun(vllm_url: str, stop, fallback: FakeSpec | None = None, rate: float = 4.0, n: int = 60,
                stop_after_s: float = 5.0, breaker: dict | None = None) -> dict:
    """Send `n` requests at `rate`/s through the chain; call `stop()` after `stop_after_s` seconds (it stops vLLM).

    Returns the served-by line in send order (v = vLLM, a = the fallback, x = failed), the requests that reached
    vLLM after the stop, the times the breaker opened, and the bench result."""
    s = vllm_stack(vllm_url, fallback, {"breaker": breaker or {"failure_threshold": 3, "recovery_timeout_s": 5}})
    try:
        k = s.issue_key("team-a")
        timer = threading.Timer(stop_after_s, stop)
        timer.start()
        t0 = time.monotonic()
        try:
            run = bench.run(s.url, [bench.TenantScript("team-a", k, rate=rate, n=n, max_tokens=16)], seed=1)
        finally:
            timer.cancel()
        order = sorted(run.samples, key=lambda x: x.t_send)
        line = "".join({VLLM_TARGET: "v", "acme/fast": "a"}.get(x.target, "x") for x in order)
        reached_dead = sum(1 for d in s.decisions() for a in d.get("attempts", [])
                           if a["target"] == VLLM_TARGET and a["outcome"] == "fallthrough")
        opens = s.call(lambda: s.gateway.router.breakers[VLLM_TARGET].opens)
        return {"line": line, "reached_dead": reached_dead, "breaker_opens": opens, "run": run,
                "elapsed_s": time.monotonic() - t0}
    finally:
        s.stop()


# --------------------------------------------------------------------------------------------- notebook 03
def prefix_cache(vllm_url: str, system: str, fallback: FakeSpec | None = None, overrides: dict | None = None) -> list:
    """Three requests sharing a long system prompt -- team-a twice, then team-b -- with a per-run nonce in the prompt,
    so blocks cached by an earlier run cannot hit. Returns (who, label, prompt tokens, cached tokens, TTFT s)."""
    nonce = f"[run {uuid.uuid4().hex[:12]}] "
    s = vllm_stack(vllm_url, fallback, overrides)
    try:
        keys = {"team-a": s.issue_key("team-a"), "team-b": s.issue_key("team-b")}
        rows = []
        for who, q in (("team-a", "first"), ("team-a", "second"), ("team-b", "first")):
            r = s.chat(keys[who], [{"role": "system", "content": nonce + system}, {"role": "user", "content": q}],
                       stream=True, stream_options={"include_usage": True}, max_completion_tokens=8)
            u = r.usage or {}
            rows.append((who, label(r), u.get("prompt_tokens"), (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                         r.ttft_s))
        return rows
    finally:
        s.stop()


def check_prefix_rows(rows: list, block: int = 16) -> None:
    """The rule a real vLLM must follow, asserted only on the rows it served: full blocks only, never the last token;
    a cold first request; a warm second one inside the tenant; nothing shared with another tenant's salt."""
    measured = {i: r for i, r in enumerate(rows) if r[1] == "MEASURED" and r[3] is not None}
    for _, _, n, hit, _ in measured.values():
        assert hit % block == 0 and hit <= (n - 1) // block * block, (n, hit)
    if 0 in measured:
        assert measured[0][3] == 0, "team-a's first request hit blocks it never computed (a stale run, or no nonce)"
    if 1 in measured:
        assert measured[1][3] > 0, "team-a's second request found none of its own blocks (prefix caching off?)"
    if 2 in measured:
        assert measured[2][3] == 0, "team-b hit team-a's blocks: the gateway did not send a per-tenant cache_salt"


# --------------------------------------------------------------------------------------------- notebook 04
def scrape(vllm_url: str) -> dict | None:
    """vLLM's token counters, or None if /metrics does not answer."""
    try:
        m = promtext.parse(client.get_text(vllm_url.rstrip("/") + "/metrics", timeout=5))
    except OSError:
        return None
    return {k: promtext.total(m, k) for k in ("vllm:prompt_tokens_total", "vllm:generation_tokens_total")}


def reconcile(vllm_url: str, n: int = 20, fallback: FakeSpec | None = None) -> dict | None:
    """Scrape /metrics, send `n` requests through a gateway whose first target is vLLM, scrape again, and reconcile the
    ledger's vLLM rows with the counters' differences. None if /metrics is unreachable (then T1 is skipped)."""
    before = scrape(vllm_url)
    if before is None:
        return None
    s = vllm_stack(vllm_url, fallback)
    try:
        k = s.issue_key("team-a")
        run = bench.run(s.url, [bench.TenantScript("team-a", k, rate=4, n=n, max_tokens=64)], seed=1)
        time.sleep(0.2)
        after = scrape(vllm_url)
        if after is None:
            return None
        delta = {k2: after[k2] - before[k2] for k2 in before}
        rec = metering.reconcile(s.ledger(), {"local": {"prompt_tokens": delta["vllm:prompt_tokens_total"],
                                                        "generation_tokens": delta["vllm:generation_tokens_total"]}})
        served = sum(1 for x in run.samples if x.target == VLLM_TARGET)
        return {"reconcile": rec, "served_by_vllm": served, "sent": len(run.samples)}
    finally:
        s.stop()
