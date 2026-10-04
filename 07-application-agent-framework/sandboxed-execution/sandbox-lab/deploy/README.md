# deploy — the same sandbox at three places on the isolation ladder

Each target runs the same `sandboxlab/wrapper.py` (budgets, truncated output, one exit reason) and the
same egress proxy. Only the boundary around them changes.

| Target | Boundary | Needs | Cost | Tier |
|---|---|---|---|---|
| [`docker/`](docker/) | a hardened container (runc), with gVisor (`runsc`) as an option. The proxy is on a Unix socket | Docker. gVisor needs Linux + sudo | $0 | T0 + Docker |
| [`kind/`](kind/) | pod per execution on Kubernetes 1.34: restricted Pod Security, default-deny NetworkPolicy, admission policy. It has **no gVisor** | Docker, kind, kubectl | $0 | T0 + Docker |
| [`gcp/terraform/`](gcp/terraform/) + [`gke/`](gke/) | the same manifests on a GKE Sandbox (gVisor) node pool, private nodes, no NAT | a GCP project with billing | pay per use (see [gcp/README.md](gcp/README.md)) | T3 |

*T0 = laptop or Colab CPU, free. T3 = the Google Cloud deployment, optional.* No part of this lab needs a GPU.

Each script prints each command before it runs that command. Each script also obeys `DRY_RUN=1` (print,
run nothing). Thus you can read any procedure on a machine without Docker or a cloud account:

```bash
DRY_RUN=1 deploy/docker/run-hardened.sh deploy/docker/examples/fetch_weather.py
DRY_RUN=1 deploy/kind/up.sh
DRY_RUN=1 deploy/gke/apply.sh
```

The YAML under `kind/` and `gke/` and `docker/seccomp-sandbox.json` come from
`sandboxlab/k8s/policy.py` and `sandboxlab/seccomp.py`. Edit those two files, then run
`python3 tools/render_manifests.py`. A test fails if a generated file no longer agrees with its source.
[`versions.env`](versions.env) pins the versions in one place.

**Cost.** Docker and kind cost $0 on your machine. For GKE, see [gcp/README.md](gcp/README.md).

**Clean up.** For kind, run `deploy/kind/down.sh`. For GKE, run `terraform -chdir=deploy/gcp/terraform destroy`.
The Docker target leaves nothing that continues to run (`--rm`). `run-with-proxy.sh` stops its proxy and
stub on exit.
