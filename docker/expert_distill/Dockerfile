# TempoRAL Expert Distill - Docker for RTX 4090 experiment
#
# Build:
#   docker build -t temporal-expert-distill -f temporal/docker/expert_distill/Dockerfile .
#
# Run:
#   docker compose -f temporal/docker/expert_distill/docker-compose.yml up

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    apt-transport-https \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install pip for python3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Install Google Cloud SDK (for gsutil to download data/checkpoints)
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------

# PyTorch (CUDA 12.4)
RUN pip install --no-cache-dir \
    torch==2.7.1 --index-url https://download.pytorch.org/whl/cu124

# JAX (CUDA 12, needed for loading orbax checkpoint)
RUN pip install --no-cache-dir \
    "jax[cuda12]==0.5.3"

# Core dependencies
RUN pip install --no-cache-dir \
    "numpy>=1.24.0,<2.0.0" \
    "flax==0.10.2" \
    "orbax-checkpoint>=0.6.0,<0.12.0" \
    matplotlib \
    scikit-learn \
    pyyaml \
    tqdm

# TensorFlow (CPU only - for DROID dataset loading via tfds)
RUN pip install --no-cache-dir \
    "tensorflow-cpu==2.15.0" \
    "tensorflow-datasets==4.9.9"

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
WORKDIR /app

# Copy the training script
COPY temporal/docker/expert_distill/run_expert_distill.py /app/run_expert_distill.py
COPY temporal/docker/expert_distill/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create directories for data, checkpoints, and output
RUN mkdir -p /dataset /checkpoint /output

ENTRYPOINT ["/app/entrypoint.sh"]
