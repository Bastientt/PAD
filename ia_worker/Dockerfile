# On utilise une image un peu plus robuste que la "slim"
FROM python:3.11-bullseye

ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

# Installation des dépendances système indispensables
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Mise à jour de pip et installation des versions "Gold" (stables ensemble)
RUN pip install --upgrade pip
RUN pip install protobuf==3.20.3
RUN pip install mediapipe==0.10.11
RUN pip install opencv-python-headless numpy==1.24.3 redis boto3 scikit-learn

COPY . .

# On s'assure qu'aucun fichier ne s'appelle mediapipe.py
RUN rm -f mediapipe.py

CMD ["python", "-u", "ai_worker.py"]