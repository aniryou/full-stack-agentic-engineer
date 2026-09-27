"""scalelab — the core concepts of scaling an agentic solution, in about a thousand lines.

One module per idea, each readable top to bottom:

    capacity.py    the arithmetic: rates → tokens/min → Little's law → PT units → dollars (Gemini)
    mistral.py     the same arithmetic on Mistral's API, plus a self-hosted vLLM fleet: its size, its
                   price, and the GPU price at which it beats the API (the Mistral provider)
    serving.py     what one self-hosted vLLM replica delivers: KV memory → concurrency,
                   HBM bandwidth → time per token, batch ↔ throughput
    clock.py       a virtual clock so simulations run 50× faster than real time (or in pure virtual time)
    model.py       a fake model on three backends — a hosted API (a shared limit that answers 429),
                   a fleet of vLLM replicas (that queue and slow down), and the hybrid of the two —
                   with prefix caching; plus the real calls for reference
    resilience.py  token bucket, backoff with jitter, circuit breaker, call_with_retries
    admission.py   degrade levels and the in-flight cap — the brake on the overload loop
    tools.py       simulated CRM / billing systems with QPS limits
    loop.py        the agent turn: budgets, checkpoints, idempotent tools, resume
    sim.py         a load generator that runs many users against any backend and reports latency,
                   shedding, spill-over and cost

The backend and provider are switches: ``sim.make_setup(mode, provider=...)``, or the environment
variables ``SCALELAB_BACKEND`` (hosted | local | hybrid) and ``SCALELAB_PROVIDER`` (gemini | mistral)
when ``make_setup`` is called without them. Unset, everything is the Google Cloud anchor scenario:
a hosted Gemini pool.

Nothing here needs cloud credentials, a key or a GPU. Where each piece lives on Google Cloud is in
docs/04-gcp-mapping.md; the self-hosted fleet on Kubernetes is in its last section.
"""

__version__ = "0.3.0"
