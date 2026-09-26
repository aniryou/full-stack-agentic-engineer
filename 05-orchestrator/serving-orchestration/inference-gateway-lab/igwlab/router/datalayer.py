"""The data layer: what the router knows about each endpoint, and how fresh it is.

The one idea: routing signals come from two places with very different freshness.

* **Scraped engine metrics** (`/metrics`, vLLM names): waiting queue, running requests, KV-cache
  usage, loaded LoRA adapters, cache geometry. Accurate but *lagged* by the scrape interval
  (llm-d's default base tick is 50 ms) — a burst of requests arriving inside one interval all
  see the same stale queue depth and can pile onto the same "idle" replica (the herd).
* **Router-local in-flight counters**: requests and prompt tokens the router itself has sent
  to an endpoint and not yet seen finish. Instant, but blind to work that did not come through
  this router (another router replica, retries, direct traffic).

The EPP's `core-metrics-extractor` maps these vLLM series to its attributes:

    vllm:num_requests_waiting -> WaitingQueueSize      vllm:kv_cache_usage_perc -> KVCacheUsagePercent
    vllm:num_requests_running -> RunningRequestsSize   vllm:lora_requests_info  -> Active/WaitingModels
    vllm:cache_config_info    -> block size, number of GPU KV blocks

For each metric the EPP (llm-d-router v0.10.0, `getLatestMetric`) keeps ONE series: the one with
the latest timestamp -- vLLM's exposition carries no timestamps, so in practice the first
matching series. It does not aggregate. With one engine per pod that is the whole story; a pod
that exposes several data-parallel ranks (`engine="0"`, `engine="1"`, ...) is read as rank 0
only. The lab router sums the ranks' queues and takes the maximum KV usage instead (its own
choice, `aggregate="sum"`); `extract_vllm(fam, aggregate="first")` reproduces the EPP.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from ..promtext import Families

__all__ = ["EndpointMetrics", "Endpoint", "Datastore", "extract_vllm", "Scraper"]


@dataclass
class EndpointMetrics:
    waiting: int = 0
    running: int = 0
    kv_usage: float = 0.0                                   # 0..1
    active_models: set = field(default_factory=set)         # LoRA adapters in running requests
    waiting_models: set = field(default_factory=set)        # LoRA adapters in waiting requests
    max_active_models: int = 0                              # vLLM --max-loras
    block_size: int = 0                                     # engine KV block size (tokens)
    num_gpu_blocks: int = 0                                 # engine KV capacity (blocks)
    update_time: float = 0.0                                # monotonic seconds; 0 = never scraped

    @property
    def fresh(self) -> bool:
        return self.update_time > 0


def _split(v: str | None) -> set:
    return {x for x in (v or "").split(",") if x}


def extract_vllm(fam: Families, aggregate: str = "sum") -> EndpointMetrics:
    """Map one scrape of a vLLM (or vLLM-compatible) `/metrics` page to routing attributes.

    aggregate="sum" (the lab router's choice): queue/running are summed across the `engine`
    label (data-parallel ranks) and KV usage is the maximum (the most constrained rank).
    aggregate="first" (what the EPP does): the first series of each metric, nothing summed.
    `vllm:gpu_cache_usage_perc` is the pre-V1 name of the KV gauge and is accepted as a fallback.
    """
    if aggregate not in ("sum", "first"):
        raise ValueError("aggregate must be 'sum' or 'first'")
    total = fam.sum if aggregate == "sum" else fam.value          # Families.value = the first series
    worst = fam.max if aggregate == "sum" else fam.value
    m = EndpointMetrics()
    m.waiting = int(total("vllm:num_requests_waiting", 0))
    m.running = int(total("vllm:num_requests_running", 0))
    kv = worst("vllm:kv_cache_usage_perc")
    if kv is None:
        kv = worst("vllm:gpu_cache_usage_perc", 0.0)
    m.kv_usage = float(kv)
    lora = fam.get("vllm:lora_requests_info")
    if lora:
        latest = max(lora, key=lambda s: s.value)          # the gauge value is a timestamp
        m.active_models = _split(latest.label("running_lora_adapters"))
        m.waiting_models = _split(latest.label("waiting_lora_adapters"))
        m.max_active_models = int(float(latest.label("max_lora") or 0))
    info = fam.get("vllm:cache_config_info")
    if info:
        m.block_size = int(float(info[0].label("block_size") or 0))
        m.num_gpu_blocks = int(float(info[0].label("num_gpu_blocks") or 0))
    return m


@dataclass(eq=False)          # identity semantics: an endpoint is a thing, not a value
class Endpoint:
    name: str
    url: str                                   # base URL, e.g. http://127.0.0.1:8001
    labels: dict = field(default_factory=dict)
    metrics: EndpointMetrics = field(default_factory=EndpointMetrics)
    inflight_requests: int = 0                 # routed here, response not finished
    inflight_tokens: int = 0                   # uncached prompt tokens routed here, first token not yet seen
    scrape_errors: int = 0
    routed_total: int = 0


class Datastore:
    """The endpoint set the scheduler sees (in Kubernetes: the pods an InferencePool selects)."""

    def __init__(self, endpoints=()):
        self._eps: dict[str, Endpoint] = {}
        for e in endpoints:
            self.add(e)

    def add(self, ep: Endpoint) -> Endpoint:
        self._eps[ep.name] = ep
        return ep

    def remove(self, name: str) -> None:
        self._eps.pop(name, None)

    def get(self, name: str) -> Endpoint:
        return self._eps[name]

    def list(self) -> list[Endpoint]:
        return [self._eps[k] for k in sorted(self._eps)]

    def __len__(self):
        return len(self._eps)


class Scraper:
    """Poll every endpoint's `/metrics` on a fixed interval and store the extracted attributes."""

    def __init__(self, datastore: Datastore, session, interval_s: float = 0.05, path: str = "/metrics",
                 timeout_s: float = 1.0, clock=time.monotonic):
        self.ds, self.session = datastore, session
        self.interval_s, self.path, self.timeout_s, self.clock = interval_s, path, timeout_s, clock
        self._task: asyncio.Task | None = None
        self.scrapes = 0

    async def scrape_once(self, ep: Endpoint) -> bool:
        import aiohttp
        try:
            async with self.session.get(ep.url + self.path, timeout=aiohttp.ClientTimeout(total=self.timeout_s)) as r:
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}")
                text = await r.text()
            m = extract_vllm(Families.from_text(text))
            m.update_time = self.clock()
            ep.metrics = m
            ep.scrape_errors = 0
            self.scrapes += 1
            return True
        except Exception:                      # keep the last good sample; count the failure
            ep.scrape_errors += 1
            return False

    async def refresh(self) -> None:
        await asyncio.gather(*(self.scrape_once(e) for e in self.ds.list()))

    async def _loop(self):
        while True:
            await self.refresh()
            await asyncio.sleep(self.interval_s)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
