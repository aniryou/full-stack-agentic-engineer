# deploy — the same sandbox at three places on the isolation ladder

Every target runs the same `sandboxlab/wrapper.py` (budgets, truncated output, one exit reason) and the
same egress proxy; what changes is the boundary around them.

| Target | Boundary | Needs | Cost | Tier |
|---|---|---|---|---|
| [`docker/`](docker/) | a hardened container (runc), optionally gVisor (`runsc`); the proxy on a Unix socket | Docker; gVisor needs Linux + sudo | $0 | T0 + Docker |
| [`kind/`](kind/) | pod per execution on Kubernetes 1.34: restricted Pod Security, default-deny NetworkPolicy, admission policy; **no gVisor** | Docker, kind, kubectl | $0 | T0 + Docker |
| [`gcp/terraform/`](gcp/terraform/) + [`gke/`](gke/) | the same manifests on a GKE Sandbox (gVisor) node pool, private nodes, no NAT | a GCP project with billing | pay per use (see [gcp/README.md](gcp/README.md)) | T3 |

*T0 = laptop or Colab CPU, free; T3 = the Google Cloud deployment, optional.* Nothing in this lab needs a GPU.

Every script prints each command before running it and honours `DRY_RUN=1` (print, run nothing), so you
can read any procedure on a machine without Docker or a cloud account:

```bash
DRY_RUN=1 deploy/docker/run-hardened.sh deploy/docker/examples/fetch_weather.py
DRY_RUN=1 deploy/kind/up.sh
DRY_RUN=1 deploy/gke/apply.sh
```

The YAML under `kind/` and `gke/` and `docker/seccomp-sandbox.json` are generated from
`sandboxlab/k8s/policy.py` and `sandboxlab/seccomp.py`: edit those, then `python3 tools/render_manifests.py`
(a test fails if a generated file drifts). Versions are pinned once, in [`versions.env`](versions.env).

**Cost.** Docker and kind: $0 on your machine. GKE: see [gcp/README.md](gcp/README.md).

**Clean up.** `deploy/kind/down.sh`; `terraform -chdir=deploy/gcp/terraform destroy`. The Docker target leaves
nothing running (`--rm`); `run-with-proxy.sh` stops its proxy and stub on exit.
