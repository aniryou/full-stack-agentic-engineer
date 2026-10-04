# deploy/docker — one execution in a hardened container, and the proxy as its only way out

**What it does.** `run-hardened.sh` runs a Python file (or stdin) in `python:3.12-slim` through the
wrapper of the lab, and prints one `@@SANDBOX_RESULT@@` JSON line. All the controls from notebook 01 are
on:

- `--network none`, `--read-only`, `--cap-drop ALL`, `no-new-privileges`,
- the seccomp profile of the lab,
- `--pids-limit 64`, `--memory 320m` without swap, `--cpus 1`,
- UID 65534,
- `noexec` tmpfs mounts with a size limit,
- `--init`.

`RUNTIME=runsc` adds gVisor. `run-with-proxy.sh` also starts the egress proxy on a Unix socket and the
stand-in API. It gives the container only the socket, through a bind mount. Thus the container has no
network, one route out, and no key.

**Cost.** $0. The image is ~50 MB compressed (verify). gVisor is a ~30 MB package (verify).

**Clean up.** There is nothing to clean. Containers run with `--rm`. `run-with-proxy.sh` stops its proxy
and stub on exit. It leaves only `/tmp/sandboxlab-egress/` (a token file that you can read). To remove
the directory, run `rm -rf /tmp/sandboxlab-egress`.

**Needs.** You need Docker Engine or Docker Desktop. For gVisor (`install-gvisor.sh`), you need Linux 5.6+,
x86_64 or arm64, and Debian/Ubuntu with systemd and sudo. The gVisor runtime does not operate
in the VM of Docker Desktop or on Colab.

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
| `run-hardened.sh` | the hardened `docker run`. It has the same flags as `sandboxlab.docker.DockerSandbox.hardened()`, and a test keeps them the same |
| `run-with-proxy.sh` | the host proxy on `/tmp/sandboxlab-egress/proxy.sock`, the stub on `127.0.0.1:8081`, and the container with only the socket |
| `install-gvisor.sh` | the apt route of gVisor, then `runsc install`, which registers the runtime `runsc` in `/etc/docker/daemon.json` |
| `seccomp-sandbox.json` | a generated file. It is the default profile of Docker minus `memfd_create`, `memfd_secret`, `execveat`, `socketcall`, the ptrace rule, and socket families other than `AF_UNIX`/`AF_INET`/`AF_INET6` (see `sandboxlab/seccomp.py`) |
| `examples/fetch_weather.py` | code that calls the API through `$SANDBOX_PROXY_URL` and makes sure that the sandbox holds no key |

## Notes

* **Why a socket, not a network.** `--network none` leaves only a loopback interface. A path-based Unix
  socket is a file. Thus a bind mount gives the container exactly one peer. The alternative is an
  `--internal` Docker network, with the proxy attached to it and to the bridge. This alternative also
  works, but it lets the sandbox reach every other container on that network.
* **`--pids-limit`, not `--ulimit nproc`.** `nproc` counts processes for each user across the full host.
  All the containers that run as 65534 share that count. The pids cgroup counts the tasks of this
  container. Under gVisor, the host cgroup sees the threads of the Sentry. Thus the wrapper also sets
  `RLIMIT_NPROC`, which the gVisor kernel counts for each sandbox (verify).
* **gVisor and limits.** gVisor does not enforce cgroup limits *inside* the sandbox. The flags of Docker
  set them on the host cgroup of the sandbox. Also, with gVisor, code can still use what the sandbox
  gets: the socket, the workspace and the allowlisted routes of the proxy.
* **The seccomp profile** starts from moby/profiles `seccomp/default.json` (Apache-2.0). The lab got
  that file on 2026-09-26 and bundles it as `sandboxlab/data/moby-seccomp-default.json`.
