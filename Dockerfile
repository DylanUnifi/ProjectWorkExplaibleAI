# ── Base image: PyTorch + CUDA 11.8 (cpu-only fallback still works) ──────────
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Project source ────────────────────────────────────────────────────────────
COPY . .

# ── Default command (override at runtime) ────────────────────────────────────
CMD ["python", "train_clevr_hans.py", "--config", "configs/clevr_hans.yaml"]