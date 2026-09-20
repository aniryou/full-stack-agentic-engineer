"""scalelab — the core concepts of scaling an agentic solution on Mistral models, in ~900 lines.

One module per idea, each readable top to bottom:

    capacity.py    the arithmetic: rates → tokens/min → Little's law → the API's rate limit and
                   price, or a GPU fleet and its price → where one beats the other → what breaks first
    serving.py     what one self-hosted vLLM replica delivers: KV memory → concurrency,
                   HBM bandwidth → time per token, batch ↔ throughput
    clock.py       a virtual clock so simulations run 50× faster than real time
    model.py       a fake Mistral model on three backends — the API (a shared limit that answers
                   429), a fleet of replicas (that queue and slow down), and the hybrid of the two —
                   plus the real calls (mistralai SDK; vLLM's OpenAI-compatible endpoint)
    resilience.py  token bucket, backoff with jitter, circuit breaker, call_with_retries
    admission.py   degrade levels and the in-flight cap — the brake on the overload loop
    tools.py       simulated CRM / billing systems with QPS limits
    loop.py        the agent turn: budgets, checkpoints, idempotent tools, resume
    sim.py         a load generator that runs many users against any backend and reports
                   latency, shedding, spill-over and cost

Nothing here needs cloud credentials or a GPU. Where each piece lives on Kubernetes (and on
Azure, AWS and Google Cloud) is in docs/04-platform-mapping.md.
"""

__version__ = "0.3.0"
