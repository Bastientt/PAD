# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — AI Video Detector
# ─────────────────────────────────────────────────────────────────────────────
# Build :  docker build -t ai-video-detector .
# Run   :  docker run --rm -v $(pwd)/videos:/data/input ai-video-detector analyze /data/input/video.mp4
# Tests :  docker run --rm ai-video-detector test
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

LABEL maintainer="ai-video-detector"
LABEL description="Détecteur de vidéos générées par IA — analyse FFT + provenance C2PA"

# Dépendances système pour OpenCV et ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ─── Dépendances Python ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Code source ─────────────────────────────────────────────────────────────
COPY ai_video_detector.py    .
COPY generate_test_videos.py .
COPY test_detector.py        .
COPY entrypoint.sh           .
RUN chmod +x entrypoint.sh

# Dossiers de données
RUN mkdir -p /data/input /data/output /data/corpus

# Variable d'environnement pour désactiver les avertissements OpenCV
ENV OPENCV_LOG_LEVEL=ERROR
ENV PYTHONUNBUFFERED=1

# Utilisateur non-root
RUN useradd -m -u 1000 detector
RUN chown -R detector:detector /app /data
USER detector

VOLUME ["/data/input", "/data/output", "/data/corpus"]

ENTRYPOINT ["./entrypoint.sh"]
CMD ["--help"]