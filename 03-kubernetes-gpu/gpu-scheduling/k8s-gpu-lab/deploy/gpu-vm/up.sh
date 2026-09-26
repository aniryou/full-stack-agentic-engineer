#!/usr/bin/env bash
# T1/T2: a real one-node Kubernetes on a GPU VM you control (Lambda, a GCP VM, your own box) -
# k3s + the NVIDIA device plugin + GPU Feature Discovery, optionally with time-slicing.
# Run it ON the VM, as a user with sudo. The VM needs the NVIDIA driver and the NVIDIA Container
# Toolkit already installed (Lambda Stack and GCP Deep Learning VM images ship both - verify).
#   deploy/gpu-vm/up.sh                              # whole GPUs
#   TIME_SLICING_REPLICAS=4 deploy/gpu-vm/up.sh      # each GPU advertised as 4 nvidia.com/gpu
#   DRY_RUN=1 deploy/gpu-vm/up.sh                    # print every command, run none
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib.sh
. "$HERE/../lib.sh"
# shellcheck source=../versions.env
. "$HERE/../versions.env"

REPLICAS="${TIME_SLICING_REPLICAS:-1}"
KCFG=/etc/rancher/k3s/k3s.yaml
SUDO=""
[[ "$(id -u)" == "0" ]] || SUDO="sudo"
export KUBECONFIG="$KCFG"

need nvidia-smi "install the NVIDIA driver first (or start from a GPU VM image that has it)"
need nvidia-container-runtime "install the NVIDIA Container Toolkit: https://github.com/NVIDIA/nvidia-container-toolkit"
need curl
need helm "https://helm.sh/docs/intro/install/"
k() { run kubectl --kubeconfig "$KCFG" "$@"; }

log "the GPU(s) the driver sees - this is what the device plugin will advertise"
run nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv

log "k3s from channel $K3S_CHANNEL (it detects the NVIDIA runtime and registers RuntimeClass 'nvidia')"
run sh -c "curl -sfL https://get.k3s.io | INSTALL_K3S_CHANNEL=$K3S_CHANNEL sh -s - --write-kubeconfig-mode 644"
retry 20 6 kubectl --kubeconfig "$KCFG" wait --for=condition=Ready node --all --timeout=60s
run $SUDO grep -c nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml || \
  die "k3s found no NVIDIA runtime: install the NVIDIA Container Toolkit, then: sudo systemctl restart k3s"
k get runtimeclass nvidia

log "NVIDIA device plugin $NVDP_CHART_VERSION + GPU Feature Discovery (pods use RuntimeClass nvidia)"
args=(upgrade -i nvdp nvdp/nvidia-device-plugin --namespace nvidia-device-plugin --create-namespace
      --version "$NVDP_CHART_VERSION" --set runtimeClassName=nvidia --set gfd.enabled=true --wait --timeout 5m)
if [[ "$REPLICAS" -gt 1 ]]; then
  cfg="$(mktemp "${TMPDIR:-/tmp}/nvdp-config.XXXXXX")"
  trap 'rm -f "$cfg"' EXIT
  # the device plugin's own config format (k8sgpu.gpuvm.TIME_SLICING_CONFIG; a test keeps them equal)
  cat >"$cfg" <<EOF
version: v1
sharing:
  timeSlicing:
    resources:
    - name: nvidia.com/gpu
      replicas: $REPLICAS
EOF
  log "time-slicing: each GPU becomes $REPLICAS nvidia.com/gpu (config below; no isolation between the pods)"
  cat "$cfg"
  args+=(--set-file "config.map.config=$cfg")
fi
run helm repo add nvdp https://nvidia.github.io/k8s-device-plugin --force-update
run helm repo update nvdp
run helm "${args[@]}"

log "what Kubernetes now sees: allocatable nvidia.com/gpu and the GFD labels"
if [[ "$DRY_RUN" != "1" ]]; then
  for _ in $(seq 1 30); do
    got="$(kubectl --kubeconfig "$KCFG" get nodes -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}')"
    [[ -n "$got" && "$got" != "0" ]] && break
    sleep 5
  done
  [[ -n "${got:-}" && "$got" != "0" ]] || die "no nvidia.com/gpu allocatable yet: kubectl -n nvidia-device-plugin get pods, then read the device plugin pod's logs"
fi
k get nodes -o custom-columns='NAME:.metadata.name,GPUS:.status.allocatable.nvidia\.com/gpu,PRODUCT:.metadata.labels.nvidia\.com/gpu\.product,REPLICAS:.metadata.labels.nvidia\.com/gpu\.replicas,MEM_MIB:.metadata.labels.nvidia\.com/gpu\.memory'

log "smoke test: one GPU through the real device plugin"
k apply -f "$HERE/10-smoke.yaml"
k wait --for=condition=complete job/gpu-smoke --timeout=300s
k logs job/gpu-smoke
echo "next: kubectl apply -f $HERE/20-time-sliced.yaml ; kubectl get pods -o wide   (see README.md)"
