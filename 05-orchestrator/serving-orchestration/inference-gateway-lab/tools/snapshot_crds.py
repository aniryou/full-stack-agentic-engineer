#!/usr/bin/env python3
"""Snapshot upstream CRD schemas into igwlab/crds/*.json (descriptions stripped, one version).

Maintainer tool (needs the upstream CRD YAML files locally, e.g. from a git checkout):

    python3 tools/snapshot_crds.py <crd.yaml> <version> "<source note>" [<crd.yaml> <version> "<note>" ...]

The snapshots let tests and notebook 05 validate this lab's manifests offline against the
real openAPIV3Schema (types, required fields, enums, patterns, bounds, unknown fields).
Snapshots in this repo (Sep 2026):
  InferencePool v1        kubernetes-sigs/gateway-api-inference-extension v1.6.2
  InferenceObjective v1alpha2  llm-d/llm-d-router v0.10.0
  Gateway, HTTPRoute v1   kubernetes-sigs/gateway-api v1.6.2 (standard channel)
  PodMonitoring v1        GoogleCloudPlatform/prometheus-engine main@6585722 (2026-09-25)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

OUT = Path(__file__).resolve().parents[1] / "igwlab" / "crds"


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k != "description"}
    if isinstance(o, list):
        return [strip(x) for x in o]
    return o


def snapshot(path: str, version: str, note: str) -> Path:
    crd = yaml.safe_load(Path(path).read_text())
    spec = crd["spec"]
    ver = next(v for v in spec["versions"] if v["name"] == version)
    doc = {"_source": note, "group": spec["group"], "version": version, "kind": spec["names"]["kind"],
           "plural": spec["names"]["plural"], "scope": spec["scope"],
           "schema": strip(ver["schema"]["openAPIV3Schema"])}
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{spec['names']['plural']}.{spec['group']}.{version}.json"
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return out


def main(argv):
    args = argv[1:]
    if len(args) % 3 or not args:
        print(__doc__)
        return 2
    for i in range(0, len(args), 3):
        print("wrote", snapshot(*args[i:i + 3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
