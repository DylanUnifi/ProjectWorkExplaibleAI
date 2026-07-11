# Dockerfile for the XAI Comparative Study
# ========================================
# Base: NVIDIA CUDA 13.0
#
# Build : docker compose build
# Run   : see docker-compose.yml

FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

# Avoid prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        git \
        unzip \
        libgl1 \
        libglib2.0-0 \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Alias python to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# ── Python dependencies ───────────────────────────────────────────────
ENV PIP_BREAK_SYSTEM_PACKAGES=1
COPY requirements.txt .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 && \
    pip install -r requirements.txt

# ── Workspace ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Source code ────────────────────────────────────────────────────────
COPY . .

# ── Output directories ─────────────────────────────────────────────────
RUN mkdir -p checkpoints

# Set Python path so imports work correctly
ENV PYTHONPATH=/app

# Default command
CMD ["bash"]
