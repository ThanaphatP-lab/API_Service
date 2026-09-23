# One CUDA-enabled image shared by every API service in docker-compose.yml.
# CUDA 12.9 + PaddlePaddle 3.3 supports Blackwell consumer GPUs (sm_120), such
# as RTX 50-series cards, in addition to prior NVIDIA GPU architectures.
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PADDLE_PDX_MODEL_SOURCE=BOS \
    MODEL_DEVICE=gpu:0

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/local/bin/python

WORKDIR /app

COPY requirements.txt ./
COPY requirements ./requirements
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . ./

# PaddleOCR/PaddleX imports OpenCV at application start; the CUDA runtime base
# image does not include OpenGL's libGL.so.1 by default.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8000
CMD ["uvicorn", "services.ocr_pipeline_paddle.main:app", "--host", "0.0.0.0", "--port", "8000"]
