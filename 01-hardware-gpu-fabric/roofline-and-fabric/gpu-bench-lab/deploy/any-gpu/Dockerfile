# gpubench on any Linux host with an NVIDIA GPU, Docker and the NVIDIA Container Toolkit.
# Build from the lab directory:   docker build -f deploy/any-gpu/Dockerfile -t gpubench .
# (deploy/any-gpu/run.sh does the build and the run for you.)
#
# VERIFY: the base image must suit both the host driver and the GPU architecture.
#   cuda12.6 (the default): R525+ drivers (CUDA 12 minor-version compatibility); kernels for
#     sm_50 through sm_90 only — Maxwell to Hopper. NO Blackwell (B200 sm_100, RTX 50xx / RTX PRO
#     6000 sm_120): there it fails with "no kernel image is available for execution on the device".
#   cuda13.0 / 13.2: R580+ drivers; kernels for sm_75 through sm_120 — Turing to Blackwell (no
#     Pascal/Volta). For Blackwell: BASE=pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime.
#   (2.14.0 is published only as cuda12.6, 13.0 and 13.2; the last cuda12.8 tag is 2.11.0.)
# Tags: hub.docker.com/r/pytorch/pytorch/tags
ARG BASE=pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime
FROM ${BASE}

WORKDIR /lab
COPY pyproject.toml README.md LICENSE ./
COPY gpubench ./gpubench
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python", "-m", "gpubench"]
CMD ["run", "--backend", "torch", "--out", "/results"]
