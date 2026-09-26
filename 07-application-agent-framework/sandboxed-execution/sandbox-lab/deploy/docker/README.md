# deploy/docker — one execution in a hardened container, and the proxy as its only way out

**What it does.** `run-hardened.sh` runs a Python file (or stdin) in `python:3.12-slim` with every control
from notebook 01 switched on — `--network none`, `--read-only`, `--cap-drop ALL`, `no-new-privileges`, the
lab's seccomp profile, `--pids-limit 64`, `--memory 320m` without swap, `--cpus 1`, UID 65534, size-capped
`noexec` tmpfs mounts, `--init` — through the lab's wrapper, and prints one `@@SANDBOX_RESULT@@` JSON line.
`RUNTIME=runsc` adds gVisor. `run-with-proxy.sh` also starts the egress proxy on a Unix socket and the
stand-in API, and bind-mounts only the socket: the container has no network, one route out, and no key.

**Cost.** $0. The image is ~50 MB compressed (verify); gVisor is a ~30 MB package (verify).

**Clean up.** Nothing to clean: containers run with `--rm`; `run-with-proxy.sh` kills its proxy and stub on
exit and leaves only `/tmp/sandboxlab-egress/` (a token file readable by you): `rm -rf /tmp/sandboxlab-egress`.

**Needs.** Docker Engine or Docker Desktop. gVisor (`install-gvisor.sh`): Linux 5.6+, x86_64 or arm64,
Debian/Ubuntu with systemd and sudo — not Docker Desktop's VM, not Colab.

## Run it

```bash
# from the lab root (07-application-agent-framework/sandboxed-execution/sandbox-lab)
DRY_RUN=1 deploy/docker/run-hardened.sh deploy/docker/examples/fetch_weather.py   # read the command
echo 'print(sum(range(10)))' | deploy/docker/run-hardened.sh                         # runc
deploy/docker/run-with-proxy.sh deploy/docker/examples/fetch_weather.py              # proxy over a socket
DRY_RUN=1 deploy/docker/install-gvisor.sh && deploy/docker/install-gvisor.sh         # once, on a Linux host
echo 'import os; print(os.uname())' | RUNTIME=runsc deploy/docker/run-hardened.sh    # the same, under gVisor
python3 -m sandboxlab probes --level docker:runc                                    # the attack probes, measured
python3 -m sandboxlab probes --level docker:runsc
python3 -m sandboxlab probes --level docker:default                                 # the naive container, for contrast
```

| File | What it is |
|---|---|
| `run-hardened.sh` | the hardened `docker run` (the same flags as `sandboxlab.docker.DockerSandbox.hardened()`; a test keeps them in step) |
| `run-with-proxy.sh` | host proxy on `/tmp/sandboxlab-egress/proxy.sock` + stub on `127.0.0.1:8081` + the container with only the socket |
| `install-gvisor.sh` | gVisor's apt route, then `runsc install` (registers the runtime `runsc` in `/etc/docker/daemon.json`) |
| `seccomp-sandbox.json` | generated: Docker's default profile minus `memfd_create`, `memfd_secret`, `execveat`, `socketcall`, the ptrace rule, and socket families other than `AF_UNIX`/`AF_INET`/`AF_INET6` (see `sandboxlab/seccomp.py`) |
| `examples/fetch_weather.py` | code that calls the API through `$SANDBOX_PROXY_URL` and checks it holds no key |

## Notes

* **Why a socket, not a network.** `--network none` leaves a loopback interface only. A path-based Unix
  socket is a file, so a bind mount gives the container exactly one peer. The alternative — an
  `--internal` Docker network with the proxy attached to it and to the bridge — works too, but lets the
  sandbox reach every other container on that network.
* **`--pids-limit`, not `--ulimit nproc`.** `nproc` counts processes per user across the whole host (every
  container running as 65534 shares it); the pids cgroup counts this container's tasks. Under gVisor the host
  cgroup sees the Sentry's threads, so the wrapper also applies `RLIMIT_NPROC`, which gVisor's kernel counts per
  sandbox (verify).
* **gVisor and limits.** gVisor does not enforce cgroup limits *inside* the sandbox; Docker's flags set them on
  the sandbox's host cgroup. gVisor also does not stop code from using what the sandbox is given — the socket,
  the workspace, the proxy's allowlisted routes.
* **The seccomp profile** is derived from moby/profiles `seccomp/default.json` (Apache-2.0), fetched 2026-09-26
  and bundled as `sandboxlab/data/moby-seccomp-default.json`.
