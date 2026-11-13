FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_DEVICE=cpu
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements/openthai.txt /app/requirements/openthai.txt
RUN pip install --no-cache-dir -r requirements/openthai.txt
COPY core /app/core
COPY shared /app/shared
COPY inference/openthai /app/inference/openthai
COPY services/ocr_openthai /app/services/ocr_openthai
CMD ["uvicorn", "services.ocr_openthai.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
