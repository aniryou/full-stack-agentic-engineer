"""A readable re-implementation of an LLM-aware endpoint picker (the llm-d EPP decision logic)
in front of N OpenAI-compatible backends. Read the modules in this order:

    tokens.py      pseudo-tokens: how a router "tokenizes" without a tokenizer
    prefix.py      block hashing + the approximate prefix index (per-endpoint LRU)
    datalayer.py   scraped vLLM metrics vs router-local in-flight counters
    plugins.py     producers, filters, scorers, pickers (upstream types and parameters)
    config.py      EndpointPickerConfig: parse, validate, inject upstream defaults
    scheduler.py   one scheduling cycle, and an explainable Decision
    server.py      the async proxy: admission, dispatch, byte-for-byte streaming, metrics
"""
from .config import ConfigError, PickerConfig, load_config, preset_names
from .datalayer import Datastore, Endpoint, EndpointMetrics, extract_vllm
from .plugins import REGISTRY, RequestCtx
from .prefix import PrefixIndex, PrefixMatch, block_hashes
from .scheduler import Decision, Scheduler
from .server import DESTINATION_HEADER, OBJECTIVE_HEADER, Router, RouterSettings
from .tokens import estimate_tokens

__all__ = ["ConfigError", "PickerConfig", "load_config", "preset_names", "Datastore", "Endpoint",
           "EndpointMetrics", "extract_vllm", "REGISTRY", "RequestCtx", "PrefixIndex", "PrefixMatch",
           "block_hashes", "Decision", "Scheduler", "Router", "RouterSettings", "DESTINATION_HEADER",
           "OBJECTIVE_HEADER", "estimate_tokens"]
