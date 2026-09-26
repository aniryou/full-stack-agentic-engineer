# gpubench on any Linux host with an NVIDIA GPU, Docker and the NVIDIA Container Toolkit.
# Build from the lab directory:   docker build -f deploy/any-gpu/Dockerfile -t gpubench .
# (deploy/any-gpu/run.sh does the build and the run for you.)
#
# VERIFY: the base image's CUDA must suit the host driver. cuda12.x images run on R525+ drivers
# (CUDA 12 minor-version compatibility) and still support Pascal/Volta GPUs; cuda13.x images need
# an R580+ driver and drop GPUs older than Turing. Tags: hub.docker.com/r/pytorch/pytorch/tags
ARG BASE=pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime
FROM ${BASE}

WORKDIR /lab
COPY pyproject.toml README.md LICENSE ./
COPY gpubench ./gpubench
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python", "-m", "gpubench"]
CMD ["run", "--backend", "torch", "--out", "/results"]
