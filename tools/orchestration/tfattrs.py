#!/usr/bin/env python3
"""List Terraform attribute paths for a google/google-beta/kubernetes/helm resource (provider schema v8.4.0 etc.).
usage: tfattrs.py <resource_type> [substring-filter ...]
   eg: tfattrs.py google_container_node_pool gpu reservation
Descriptions: add --desc to print the description of each matching attribute (truncated)."""
import json, sys, os
SCHEMA = os.environ.get("TF_SCHEMA_JSON", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tfschema", "schema.json"))  # generate: terraform providers schema -json
args = [a for a in sys.argv[1:] if a != "--desc"]; desc = "--desc" in sys.argv
if not args: print(__doc__); sys.exit(2)
res, filt = args[0], [f.lower() for f in args[1:]]
s = json.load(open(SCHEMA))["provider_schemas"]
found = None
for prov, ps in s.items():
    for kind in ("resource_schemas", "data_source_schemas"):
        if res in ps.get(kind, {}):
            found = ps[kind][res]["block"]; print(f"# {res} ({prov.split('/')[-1]} {kind})"); break
    if found: break
if not found: print("not found:", res); sys.exit(1)
def walk(b, p=""):
    for k, v in b.get("attributes", {}).items():
        yield p + k, v
    for k, v in b.get("block_types", {}).items():
        yield p + k + "{}", {"description": f"block nesting={v.get('nesting_mode')} min={v.get('min_items',0)} max={v.get('max_items','-')}"}
        yield from walk(v["block"], p + k + ".")
for path, v in walk(found):
    if filt and not any(f in path.lower() for f in filt): continue
    flags = ",".join(x for x in ("required", "optional", "computed") if v.get(x))
    line = f"{path}  [{flags}]"
    if desc and v.get("description"): line += "  -- " + v["description"].replace("\n", " ")[:220]
    print(line)
