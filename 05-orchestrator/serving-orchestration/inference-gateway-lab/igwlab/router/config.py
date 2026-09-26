"""EndpointPickerConfig: the YAML that decides how the router routes.

The one idea: routing policy is *data*. The llm-d EPP reads a document like

    apiVersion: llm-d.ai/v1
    kind: EndpointPickerConfig
    plugins:                       # instantiate plugins (type + optional name + parameters)
    - type: prefix-cache-scorer
    - type: queue-scorer
    schedulingProfiles:            # which instances run for a request, and scorer weights
    - name: default
      plugins:
      - pluginRef: prefix-cache-scorer
        weight: 3
      - pluginRef: queue-scorer
        weight: 2

and fills in what you left out (upstream rules, reproduced here):
  * no `schedulingProfiles` -> one `default` profile with every listed filter/scorer/picker;
  * a profile without a picker gets `max-score-picker`; a scorer without a weight gets 1.0;
  * a plugin that needs a data producer that is not configured gets the default producer
    (`approx-prefix-cache-producer`, `inflight-load-producer`);
  * `metrics-data-source` + `core-metrics-extractor` are always present.

The lab router runs exactly one scheduling profile (no P/D disaggregation) and only the plugin
types in plugins.REGISTRY; anything else is rejected with a message rather than ignored.
`to_upstream()` returns the document with lab-only parameters removed, ready to paste into the
Helm value `router.epp.pluginsCustomConfig` (see deploy/kind and deploy/gke).
"""
from __future__ import annotations

import copy
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .plugins import (REGISTRY, ActiveRequestScorer, ApproxPrefixCacheProducer, InFlightLoadProducer, Plugin,
                      PrefixCacheAffinityFilter, PrefixCacheScorer, TokenLoadScorer, UtilizationFilter, make_plugin)

API_VERSIONS = {"llm-d.ai/v1": None, "llm-d.ai/v1alpha1": "llm-d.ai/v1alpha1 is deprecated; use llm-d.ai/v1"}
KIND = "EndpointPickerConfig"
KNOWN_TOP = {"apiVersion", "kind", "plugins", "schedulingProfiles", "featureGates",
             "dataLayer", "requestHandler", "flowControl", "saturationDetector", "parser"}
PRESETS_DIR = Path(__file__).resolve().parent.parent / "configs"

__all__ = ["ConfigError", "Profile", "PickerConfig", "load_config", "preset_names", "parse_duration"]


class ConfigError(ValueError):
    pass


def parse_duration(v) -> float:
    """'50ms' | '1s' | '1m' | '250us' | number (seconds) -> seconds."""
    if isinstance(v, (int, float)):
        return float(v)
    m = re.fullmatch(r"\s*([0-9.]+)\s*(us|ms|s|m|h)\s*", str(v))
    if not m:
        raise ConfigError(f"bad duration {v!r}")
    n, unit = float(m.group(1)), m.group(2)
    return n * {"us": 1e-6, "ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]


@dataclass
class Profile:
    name: str
    filters: list = field(default_factory=list)          # [Filter]
    scorers: list = field(default_factory=list)          # [(Scorer, weight)]
    picker: Plugin | None = None


@dataclass
class PickerConfig:
    api_version: str
    plugins: dict                                        # name -> Plugin (after default injection)
    profile: Profile
    producers: list                                      # [Producer], in dependency order
    feature_gates: list
    scrape_interval_s: float | None
    metrics_path: str
    warnings: list
    raw: dict                                            # the document as written

    @property
    def flow_control(self) -> bool:
        return "flowControl" in self.feature_gates

    def describe(self) -> str:
        p = self.profile
        lines = [f"profile {p.name!r}:"]
        lines += [f"  filter  {f.TYPE:<32} {f.name}" for f in p.filters]
        lines += [f"  scorer  {s.TYPE:<32} {s.name}  weight={w:g}" for s, w in p.scorers]
        lines.append(f"  picker  {p.picker.TYPE:<32} {p.picker.name}")
        lines += [f"  data    {d.TYPE:<32} {d.name}" for d in self.producers]
        return "\n".join(lines)

    def to_upstream(self) -> dict:
        """The raw document minus lab-only parameters (and with the current apiVersion)."""
        doc = copy.deepcopy(self.raw)
        doc["apiVersion"] = "llm-d.ai/v1"
        for p in doc.get("plugins", []):
            cls = REGISTRY.get(p.get("type"))
            if cls and p.get("parameters"):
                for k in cls.LAB_ONLY:
                    p["parameters"].pop(k, None)
                if not p["parameters"]:
                    del p["parameters"]
        return doc

    def to_upstream_yaml(self) -> str:
        return yaml.safe_dump(self.to_upstream(), sort_keys=False)


def preset_names() -> list[str]:
    return sorted(p.stem for p in PRESETS_DIR.glob("*.yaml"))


def _read(src) -> dict:
    if isinstance(src, dict):
        return copy.deepcopy(src)
    if isinstance(src, Path) or (isinstance(src, str) and "\n" not in src and src.endswith((".yaml", ".yml"))):
        return yaml.safe_load(Path(src).read_text())
    if isinstance(src, str) and "\n" not in src and (PRESETS_DIR / f"{src}.yaml").exists():
        return yaml.safe_load((PRESETS_DIR / f"{src}.yaml").read_text())
    if isinstance(src, str):
        doc = yaml.safe_load(src)
        if isinstance(doc, dict):
            return doc
    raise ConfigError(f"cannot read an EndpointPickerConfig from {src!r} (presets: {preset_names()})")


def _needs(plugin: Plugin) -> list[tuple[str, str]]:
    """(producer type, producer name) pairs this plugin reads."""
    out = []
    p = plugin.params
    if isinstance(plugin, (PrefixCacheScorer, PrefixCacheAffinityFilter)):
        out.append((ApproxPrefixCacheProducer.TYPE, p.get("prefixMatchInfoProducerName") or ApproxPrefixCacheProducer.TYPE))
    if isinstance(plugin, (ActiveRequestScorer, TokenLoadScorer)):
        out.append((InFlightLoadProducer.TYPE, InFlightLoadProducer.TYPE))
    if isinstance(plugin, PrefixCacheAffinityFilter) and p["maxTTFTPenaltyMs"] > 0:
        out.append((InFlightLoadProducer.TYPE, p.get("inFlightLoadProducerName") or InFlightLoadProducer.TYPE))
    if isinstance(plugin, UtilizationFilter) and any(c["metric"] == "active-requests" for c in p["conditions"]):
        out.append((InFlightLoadProducer.TYPE, p.get("inFlightLoadProducerName") or InFlightLoadProducer.TYPE))
    if isinstance(plugin, InFlightLoadProducer):
        # uncached-token accounting reads prefix matches when a prefix producer exists
        pass
    return out


def load_config(src, seed: int = 0, clock=time.monotonic) -> PickerConfig:
    """Parse, validate and complete an EndpointPickerConfig (dict, YAML text, file path or preset name)."""
    doc = _read(src)
    warnings: list[str] = []
    if not isinstance(doc, dict):
        raise ConfigError("config must be a mapping")
    api = doc.get("apiVersion")
    if api not in API_VERSIONS:
        raise ConfigError(f"apiVersion must be one of {sorted(API_VERSIONS)} (got {api!r})")
    if API_VERSIONS[api]:
        warnings.append(API_VERSIONS[api])
    if doc.get("kind") != KIND:
        raise ConfigError(f"kind must be {KIND}")
    unknown = set(doc) - KNOWN_TOP
    if unknown:
        raise ConfigError(f"unknown top-level field(s): {sorted(unknown)}")
    for k in ("dataLayer", "requestHandler", "flowControl", "saturationDetector", "parser"):
        if k in doc:
            warnings.append(f"`{k}` is accepted but ignored by the lab router")
    gates = []
    for g in doc.get("featureGates") or []:
        name, _, val = str(g).partition("=")
        if val.lower() in ("", "true"):
            gates.append(name)
    for g in gates:
        if g != "flowControl":
            warnings.append(f"feature gate {g!r} ignored by the lab router")

    rng = random.Random(seed)
    plugins: dict[str, Plugin] = {}
    for i, spec in enumerate(doc.get("plugins") or []):
        if not isinstance(spec, dict) or "type" not in spec:
            raise ConfigError(f"plugins[{i}] needs a `type`")
        extra = set(spec) - {"type", "name", "parameters"}
        if extra:
            raise ConfigError(f"plugins[{i}]: unknown field(s) {sorted(extra)}")
        name = spec.get("name") or spec["type"]
        if name in plugins:
            raise ConfigError(f"duplicate plugin name {name!r}")
        try:
            plugins[name] = make_plugin(spec["type"], name, spec.get("parameters"), rng=random.Random(rng.random()),
                                        clock=clock)
        except ValueError as e:
            raise ConfigError(str(e)) from None
        if plugins[name].LAB_ONLY & set(spec.get("parameters") or {}):
            warnings.append(f"{name}: lab-only parameter(s) {sorted(plugins[name].LAB_ONLY & set(spec['parameters']))} "
                            "are stripped by to_upstream()")

    # ---- the scheduling profile
    profiles = doc.get("schedulingProfiles")
    if profiles is None:
        refs = [{"pluginRef": n} for n, p in plugins.items() if p.KIND in ("filter", "scorer", "picker")]
        profiles = [{"name": "default", "plugins": refs}]
    if len(profiles) != 1:
        raise ConfigError("the lab router runs exactly one scheduling profile (no P/D disaggregation)")
    prof_spec = profiles[0]
    profile = Profile(name=prof_spec.get("name", "default"))
    for ref in prof_spec.get("plugins") or []:
        pname = ref.get("pluginRef")
        if pname not in plugins:
            raise ConfigError(f"profile {profile.name!r} references unknown plugin {pname!r}")
        pl = plugins[pname]
        if pl.KIND == "filter":
            profile.filters.append(pl)
        elif pl.KIND == "scorer":
            w = float(ref.get("weight", 1.0))
            if w < 0:
                raise ConfigError(f"weight of {pname} must be >= 0")
            profile.scorers.append((pl, w))
        elif pl.KIND == "picker":
            if profile.picker is not None:
                raise ConfigError(f"profile {profile.name!r} has two pickers")
            profile.picker = pl
        else:
            warnings.append(f"{pname} ({pl.KIND}) in a profile has no scheduling role; ignored")
        if "weight" in ref and pl.KIND != "scorer":
            warnings.append(f"weight on non-scorer {pname} ignored")
    if profile.picker is None:
        plugins.setdefault("max-score-picker", make_plugin("max-score-picker"))
        profile.picker = plugins["max-score-picker"]

    # ---- data producers: configured ones first, then defaults for unmet needs
    needed: dict[str, str] = {}
    for pl in list(profile.filters) + [s for s, _ in profile.scorers]:
        for ptype, pname in _needs(pl):
            needed[pname] = ptype
    for pname, ptype in needed.items():
        if pname not in plugins:
            if pname != ptype:
                raise ConfigError(f"producer {pname!r} is referenced but not configured")
            plugins[pname] = make_plugin(ptype, pname, clock=clock)
        elif plugins[pname].TYPE != ptype:
            raise ConfigError(f"{pname!r} must be a {ptype}")
    order = {ApproxPrefixCacheProducer.TYPE: 0, InFlightLoadProducer.TYPE: 1}
    producers = sorted((p for p in plugins.values() if p.KIND == "producer"), key=lambda p: order.get(p.TYPE, 9))

    for t in ("metrics-data-source", "core-metrics-extractor", "single-profile-handler"):
        if not any(p.TYPE == t for p in plugins.values()):
            plugins[t] = make_plugin(t)
    src_plugin = next(p for p in plugins.values() if p.TYPE == "metrics-data-source")
    interval = src_plugin.params["interval"]
    return PickerConfig(api_version=api, plugins=plugins, profile=profile, producers=producers, feature_gates=gates,
                        scrape_interval_s=parse_duration(interval) if interval else None,
                        metrics_path=src_plugin.params["path"], warnings=warnings, raw=doc)
