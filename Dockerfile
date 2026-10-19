# syntax=docker/dockerfile:1
#
# vLLM + Ray image, multi-stage, built on NVIDIA's CUDA deep-learning base (CUDA 13.3, Ubuntu 24.04).
# Pinned versions (verify before building; these were the latest releases found at build time):
#   - NVIDIA base:  nvcr.io/nvidia/cuda-dl-base:26.07-cuda13.3-devel-ubuntu24.04
#   - vLLM:         0.29.0  (latest on PyPI at build time; check it targets the CUDA 13.x base)
#   - Ray:          2.56.1
#
# Build:   docker build -t vllm-claude .     (or ./build.sh)
# Run GPU: docker run --rm -it --gpus all --ipc=host -p 8000:8000 -p 8265:8265 vllm-claude bash

ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:26.07-cuda13.3-devel-ubuntu24.04

# ---------------------------------------------------------------------------
# Stage 1: builder — compiles/installs Python dependencies into a virtualenv.
# Build tools and pip caches stay in this stage and are not carried forward.
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder

ARG VLLM_VERSION=0.29.0
ARG RAY_VERSION=2.56.1

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Ubuntu 24.04 blocks system-wide pip installs, so dependencies go in a venv.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-venv python3-dev build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH

RUN pip install --upgrade pip setuptools wheel \
    && pip install "vllm==${VLLM_VERSION}" "ray[default]==${RAY_VERSION}"

# ---------------------------------------------------------------------------
# Stage 2: runtime — same CUDA base, but only the finished virtualenv is copied in.
# No compilers, apt lists, or pip cache from the builder.
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/root/.cache/huggingface

# Runtime system packages. Triton compiles a small CUDA helper at runtime, which needs
# gcc and the Python headers (Python.h, from python3-dev). Without them the model
# inspection step fails with "Python.h: No such file or directory".
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-dev build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# 8000: vLLM OpenAI-compatible API | 8265: Ray dashboard | 6379: Ray GCS head port
EXPOSE 8000 8265 6379

WORKDIR /workspace

CMD ["bash"]