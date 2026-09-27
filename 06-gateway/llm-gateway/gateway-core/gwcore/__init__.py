"""gwcore -- the LLM gateway in one sitting: one front door for many models.

    from gwcore import api, providers, routing, cache, ratelimit, metering, keys, guardrails, mcp_authz, otel, gateway

Read the modules in this order; each opens with the one idea it teaches (PRIMER.md, one level up, has the
section each module serves): api (§1) -> providers (§1, §5) -> routing (§2) -> cache (§3) -> ratelimit (§4)
-> metering and otel (§5) -> keys (§6) -> guardrails (§7) -> mcp_authz (§8) -> gateway (the §1 pipeline
that wires them together). Standard library + numpy, in-process, on a virtual clock: nothing here touches
the network, and every latency it prints is simulated.
"""
from . import api, cache, gateway, guardrails, keys, mcp_authz, metering, otel, providers, ratelimit, routing
from .gateway import Gateway, Result
from .providers import CATALOGUE, Clock, FakeProvider
from .routing import Router, Target

__all__ = ["api", "providers", "routing", "cache", "ratelimit", "metering", "otel", "keys", "guardrails", "mcp_authz",
           "gateway", "Gateway", "Result", "CATALOGUE", "Clock", "FakeProvider", "Router", "Target"]
__version__ = "0.1.0"
