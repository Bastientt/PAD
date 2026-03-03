FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip

# 1. OpenCV Headless (version stable)
RUN pip install --default-timeout=1000 opencv-python-headless==4.8.0.76

# 2. Dépendances de base + Matplotlib (requis par Mediapipe solutions)
RUN pip install --default-timeout=1000 \
    numpy==1.24.3 \
    protobuf==3.20.3 \
    absl-py attrs flatbuffers \
    matplotlib kiwisolver cycler pyparsing \
    redis boto3

# 3. MediaPipe sans ses dépendances lourdes (Torch/TF full)
RUN pip install mediapipe==0.10.11 --no-deps

COPY . .

CMD ["python", "-u", "ai_worker.py"]