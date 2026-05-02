# silent inference server — JEPA predator that hunts by audio.
# CPU-only PyTorch + FastAPI WebSocket. 5 selectable variants:
# 3E ep30 (canonical), JEPA (og baseline), 3F ep40/45/50.
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
        uvicorn[standard]==0.34.0

COPY world_model/ /app/world_model/
COPY envs/ /app/envs/
COPY scripts/ /app/scripts/
COPY client/ /app/client/

ENV SDL_VIDEODRIVER=dummy
ENV PYTHONUNBUFFERED=1

# CPU tuning for Hetzner CPX21 (4 vCPU shared). Locally on M3 the full CEM
# (16 samples × 2 iters) hits ~50ms; on this VPS it hit ~200ms which exceeds
# the 100ms tick budget. Halving samples + 1 iter brings it under 100ms.
# Thread cap prevents matmul contention with relay/lepong on the same host.
ENV SILENT_CEM_SAMPLES=8
ENV SILENT_CEM_ITERS=1
ENV SILENT_TORCH_THREADS=2
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2

EXPOSE 8801

# Serves the silent demo + WebSocket. All 5 variants wired in.
CMD ["python", "-m", "world_model.infer_silent_env", \
     "--host", "0.0.0.0", "--port", "8801", \
     "--jepa-ckpt",   "/app/checkpoints/silent_v1_3e_ep030.pt", \
     "--jepa-head",   "/app/checkpoints/3e_ep030_head_uniform.pt", \
     "--jepa-test",   "jepa_baseline:/app/checkpoints/silent_BASELINE_ep010_joint.pt:/app/checkpoints/silent_BASELINE_head_uniform.pt", \
     "--jepa-test",   "jepa_test1:/app/checkpoints/silent_v1_3f_ep010.pt:/app/checkpoints/3f_ep010_head_uniform.pt", \
     "--jepa-test",   "jepa_test2:/app/checkpoints/silent_v1_3f_ep015.pt:/app/checkpoints/3f_ep015_head_uniform.pt", \
     "--jepa-test",   "jepa_test3:/app/checkpoints/silent_v1_3f_ep020.pt:/app/checkpoints/3f_ep020_head_uniform.pt"]
