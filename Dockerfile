# ── Base image: PyTorch 2.7 + CUDA 12.8 (Blackwell sm_120 support) ──────────
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# ── System deps (single layer, minimal) ─────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps (cached unless requirements.txt changes) ────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Project source ──────────────────────────────────────────────────────────
COPY . .

# ── Default: interactive shell (override via docker-compose) ────────────────
CMD ["bash"]