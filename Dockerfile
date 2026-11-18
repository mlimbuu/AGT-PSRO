# Optional, vendor-neutral CPU container for Linux cloud hosts (AWS/GCP/Azure/etc.).
# This image does not require a GPU or a particular cluster module system.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-wandb.txt ./
# CPU-only PyTorch, avoiding vendor-specific CUDA dependencies.
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt
# Set --build-arg WITH_WANDB=1 when W&B tracking is needed in Docker.
ARG WITH_WANDB=0
RUN if [ "$WITH_WANDB" = "1" ]; then \
      python -m pip install --no-cache-dir -r requirements-wandb.txt; \
    fi
COPY . .
CMD ["python", "experiments/quick/run_svi.py", "--scenario", "five_node_two_risk"]
