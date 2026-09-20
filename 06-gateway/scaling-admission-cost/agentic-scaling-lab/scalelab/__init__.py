"""scalelab — the core concepts of scaling an agentic solution, in ~700 lines.

One module per idea, each readable top to bottom:

    capacity.py    the arithmetic: rates → tokens/min → Little's law → PT units → dollars
    clock.py       a virtual clock so simulations run 50× faster than real time
    model.py       a fake Gemini with realistic latency, a shared tokens-per-minute pool
                   that answers 429 when exhausted, and prefix caching; plus a 30-line
                   adapter showing the real call
    resilience.py  token bucket, backoff with jitter, circuit breaker, call_with_retries
    admission.py   degrade levels and the in-flight cap — the brake on the overload loop
    tools.py       simulated CRM / billing systems with QPS limits
    loop.py        the agent turn: budgets, checkpoints, idempotent tools, resume
    sim.py         a load generator that runs many users and reports latency and shedding

Nothing here needs cloud credentials. Where each piece lives on Google Cloud is in
docs/04-gcp-mapping.md.
"""

__version__ = "0.2.0"
