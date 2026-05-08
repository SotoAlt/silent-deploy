# silent inference server — JEPA predator that hunts by audio.
# CPU-only PyTorch + FastAPI WebSocket. Three selectable variants:
# canonical (3E ep30), baseline (JEPA og), federation (3F ep50).
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libgl1 libglib2.0-0 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# CPU torch wheel (no CUDA).
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1 torchvision==0.19.1 \
    && pip install --no-cache-dir \
        timm==1.0.11 \
        einops \
        h5py \
        librosa==0.10.2 \
        numpy \
        scipy \
        pygame==2.6.1 \
        pymunk==6.9.0 \
        opencv-python-headless \
        fastapi==0.115.6 \
        uvicorn[standard]==0.34.0 \
        httpx==0.27.2

COPY world_model/ /app/world_model/
COPY envs/ /app/envs/
COPY scripts/ /app/scripts/
COPY client/ /app/client/

ENV SDL_VIDEODRIVER=dummy
ENV PYTHONUNBUFFERED=1

# CPU thread budget for the predator's CEM planner. Tuned for the
# CPX31 (4 vCPU, 8 GB) where federated/relay/lepong sit < 4% CPU on
# average — silent is the gameplay-critical container, so it gets
# nearly the whole CPU budget. Earlier 2-thread cap (CPX21 era) was
# dropping planner latency to 200-260ms which made the predator use
# stale observations and feel "dumb". 4 threads brings it under 100ms.
# CEM samples/iters left at default (16/2); async background planning
# keeps the game tick rate decoupled from planner latency anyway.
ENV SILENT_TORCH_THREADS=4
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

EXPOSE 8801

# Serves the silent demo + WebSocket. Three variants exposed in the UI:
# - canonical (3E ep30): the current ship — beacon-free, joint-trained.
# - baseline (JEPA og): pre-3E, beacon-trained — kept as the "before" A/B.
# - federation (3F ep50): continuation training from 3E ep30 — first
#   model retrained on federation-pool data. Selected from the 3F sweep
#   as the strongest probe-fit candidate.
CMD ["python", "-m", "world_model.infer_silent_env", \
     "--host", "0.0.0.0", "--port", "8801", \
     "--jepa-ckpt",   "/app/checkpoints/silent_v1_3e_ep030.pt", \
     "--jepa-head",   "/app/checkpoints/3e_ep030_head_uniform.pt", \
     "--jepa-test",   "jepa_baseline:/app/checkpoints/silent_BASELINE_ep010_joint.pt:/app/checkpoints/silent_BASELINE_head_uniform.pt", \
     "--jepa-test",   "jepa_test3:/app/checkpoints/silent_v1_3f_ep020.pt:/app/checkpoints/3f_ep020_head_uniform.pt"]
