"""The gateway's configuration: providers, the model catalogue, aliases, tenants, limits, cache, guardrails.

The one idea: everything a gateway decides per request is data an operator can read in one file — which
providers exist and where their keys come from, which (provider, model) pairs are in the catalogue and what
they cost, which *alias* a client names and which ordered chain of targets serves it, which tenant may use
which alias under which limits. Code never names a model; configuration does (scaling primer §5.2: model ids
live in configuration, which is also how you survive a model's retirement).

`${VAR:-default}` in any string is replaced from the environment, so the same file serves the in-process
stack (ports filled in by `gwlab.stack`), Docker Compose and a real vLLM (deploy/any-gpu).
"""
from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import yaml

from .metering import PRICES, Price

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
POLICIES = ("ordered", "cheapest", "ewma_ttft", "canary")
INPUT_PLACEMENTS = ("off", "inline", "parallel", "shadow")
OUTPUT_PLACEMENTS = ("off", "full", "window", "parallel", "shadow")


def expand(value):
    """Replace ${VAR:-default} from the environment, recursively."""
    if isinstance(value, str):
        return _VAR.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    return value


@dataclass
class Provider:
    name: str
    dialect: str                                   # openai | anthropic
    base_url: str
    key: str = ""                                  # the provider credential: held by the gateway only
    region: str = "us"
    first_byte_timeout_s: float = 10.0
    total_timeout_s: float = 600.0
    cache_salt: bool = False                       # the upstream takes vLLM's per-request `cache_salt`
    otel_name: str | None = None                   # gen_ai.provider.name (default: the dialect's)


@dataclass
class Model:
    id: str                                        # the catalogue id, "<provider>/<name>"
    provider: str
    upstream: str                                  # the model name the provider expects
    context_window: int = 8192
    tools: bool = True
    price: Price = field(default_factory=lambda: Price(0.0, 0.0, 0.0))
    price_ref: str = ""                            # which dated price row (metering.PRICES) or "self-hosted"


@dataclass
class Alias:
    name: str
    targets: list
    policy: str = "ordered"
    canary_weight: float = 0.0                     # policy canary: share of traffic sent to targets[1] first


@dataclass
class Tenant:
    name: str
    aliases: list
    rpm: int = 600
    tpm: int = 200_000
    budget_usd: float | None = None
    tier: str = "standard"
    regions: list | None = None                    # residency: only providers in these regions
    cache: bool = True


@dataclass
class Limits:
    mode: str = "reserve"                          # reserve (reserve -> stream -> reconcile) | per_request
    minute_s: float = 60.0                         # seconds per rate-limit "minute" (notebooks compress it; say so)
    default_output_estimate: int = 512             # reserved when the caller sets no output cap
    hard_output_cap: int = 2048                    # the gateway never lets one response exceed this
    per_request_output_guess: int = 64             # per_request mode charges this for output, once, up front
    global_tpm: int = 0                            # a gateway-wide TPM (a provider share); 0 = off


@dataclass
class CacheCfg:
    exact: bool = True
    semantic: bool = True
    threshold: float = 0.92
    entity_guard: bool = True
    ttl_s: float = 3600.0
    dim: int = 1024
    classes: list = field(default_factory=lambda: ["faq"])              # declared shared classes that may be cached
    per_user_classes: list = field(default_factory=lambda: ["account"])  # ... namespaced by metadata.user as well


@dataclass
class GuardCfg:
    input: str = "off"
    output: str = "off"
    window_tokens: int = 16
    check_ms: float = 20.0


@dataclass
class BreakerCfg:
    failure_threshold: int = 3
    recovery_timeout_s: float = 5.0


@dataclass
class GatewayConfig:
    providers: dict
    models: dict
    aliases: dict
    tenants: dict
    limits: Limits = field(default_factory=Limits)
    cache: CacheCfg = field(default_factory=CacheCfg)
    guardrails: GuardCfg = field(default_factory=GuardCfg)
    breaker: BreakerCfg = field(default_factory=BreakerCfg)
    admin_token: str = "dev-admin-token"
    salt_secret: str = "dev-salt-secret"
    db: str = ":memory:"
    spans: str = ""
    mcp_servers: dict = field(default_factory=dict)  # name -> MCP server URL the gateway may call for agents
    source: str = ""

    def model(self, model_id: str) -> Model:
        return self.models[model_id]

    def provider_of(self, model_id: str) -> Provider:
        return self.providers[self.models[model_id].provider]


def bundled(name: str) -> Path:
    """Path of a config shipped with the lab (`gwlab/configs/<name>.yaml`)."""
    return Path(str(resources.files("gwlab").joinpath("configs", f"{name}.yaml")))


def _price(spec, model_id: str) -> tuple[Price, str]:
    if spec is None:
        return Price(0.0, 0.0, 0.0), ""
    if isinstance(spec, str):
        if spec not in PRICES:
            raise ValueError(f"model {model_id}: unknown price row {spec!r} (known: {sorted(PRICES)})")
        return PRICES[spec], spec
    if isinstance(spec, dict) and "gpu_hour_usd" in spec:
        from .metering import self_hosted_price
        return self_hosted_price(float(spec["gpu_hour_usd"]), float(spec["tokens_per_s"]),
                                 float(spec.get("utilisation", 1.0))), "self-hosted"
    if isinstance(spec, dict):
        return Price(float(spec["input"]), float(spec["output"]), float(spec.get("cached", spec["input"])),
                     spec.get("cache_write")), "custom"
    raise ValueError(f"model {model_id}: price must be a row name or a mapping")


def load_config(src="lab", overrides: dict | None = None) -> GatewayConfig:
    """Load a config from a bundled name, a YAML path, YAML text or a dict; `overrides` deep-merges on top."""
    if isinstance(src, dict):
        raw, where = copy.deepcopy(src), "<dict>"
    else:
        s = str(src)
        if "\n" in s:
            raw, where = yaml.safe_load(s), "<text>"
        else:
            path = Path(s) if (s.endswith((".yaml", ".yml")) or os.sep in s) else bundled(s)
            raw, where = yaml.safe_load(path.read_text()), str(path)
    if overrides:
        raw = _merge(raw, overrides)
    raw = expand(raw)
    gw = raw.get("gateway") or {}
    providers = {n: Provider(name=n, **{k: v for k, v in (p or {}).items()}) for n, p in (raw.get("providers") or {}).items()}
    for p in providers.values():
        p.first_byte_timeout_s, p.total_timeout_s = float(p.first_byte_timeout_s), float(p.total_timeout_s)
        p.cache_salt = _bool(p.cache_salt)
        if p.dialect not in ("openai", "anthropic"):
            raise ValueError(f"provider {p.name}: dialect must be openai or anthropic (gemini is table data only)")
    models = {}
    for mid, m in (raw.get("models") or {}).items():
        m = dict(m or {})
        price, ref = _price(m.pop("price", None), mid)
        models[mid] = Model(id=mid, provider=m.pop("provider"), upstream=m.pop("upstream"),
                            context_window=int(m.pop("context_window", 8192)), tools=_bool(m.pop("tools", True)),
                            price=price, price_ref=ref)
        if m:
            raise ValueError(f"model {mid}: unknown keys {sorted(m)}")
    aliases = {n: Alias(name=n, targets=list(a["targets"]), policy=a.get("policy", "ordered"),
                        canary_weight=float(a.get("canary_weight", 0.0))) for n, a in (raw.get("aliases") or {}).items()}
    tenants = {n: Tenant(name=n, **(t or {})) for n, t in (raw.get("tenants") or {}).items()}
    cfg = GatewayConfig(
        providers=providers, models=models, aliases=aliases, tenants=tenants,
        limits=_fill(Limits, raw.get("limits")), cache=_fill(CacheCfg, raw.get("cache")),
        guardrails=_fill(GuardCfg, raw.get("guardrails")), breaker=_fill(BreakerCfg, raw.get("breaker")),
        admin_token=str(gw.get("admin_token") or "dev-admin-token"), salt_secret=str(gw.get("salt_secret") or "dev-salt-secret"),
        db=str(gw.get("db") or ":memory:"), spans=str(gw.get("spans") or ""), mcp_servers=dict(raw.get("mcp_servers") or {}),
        source=where)
    validate(cfg)
    return cfg


def validate(cfg: GatewayConfig) -> None:
    """Every reference resolves; policies and placements are known values."""
    for m in cfg.models.values():
        if m.provider not in cfg.providers:
            raise ValueError(f"model {m.id}: unknown provider {m.provider!r}")
    for a in cfg.aliases.values():
        if not a.targets:
            raise ValueError(f"alias {a.name}: no targets")
        for t in a.targets:
            if t not in cfg.models:
                raise ValueError(f"alias {a.name}: unknown target {t!r}")
        if a.policy not in POLICIES:
            raise ValueError(f"alias {a.name}: policy must be one of {POLICIES}")
    for t in cfg.tenants.values():
        for a in t.aliases:
            if a not in cfg.aliases:
                raise ValueError(f"tenant {t.name}: unknown alias {a!r}")
    if cfg.limits.mode not in ("reserve", "per_request"):
        raise ValueError("limits.mode must be reserve or per_request")
    if cfg.guardrails.input not in INPUT_PLACEMENTS or cfg.guardrails.output not in OUTPUT_PLACEMENTS:
        raise ValueError(f"guardrails: input in {INPUT_PLACEMENTS}, output in {OUTPUT_PLACEMENTS}")


def _fill(cls, d):
    obj = cls()
    for k, v in (d or {}).items():
        if not hasattr(obj, k):
            raise ValueError(f"{cls.__name__}: unknown key {k!r}")
        cur = getattr(obj, k)
        if isinstance(cur, str) and v is False:          # YAML 1.1 reads an unquoted `off` as False
            v = "off"
        setattr(obj, k, _bool(v) if isinstance(cur, bool) else type(cur)(v) if isinstance(cur, (int, float)) and v is not None else v)
    return obj


def _bool(v) -> bool:
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")


def _merge(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = _merge(a.get(k), v) if k in a else copy.deepcopy(v)
        return out
    return copy.deepcopy(b)
