# RunPod Serverless: Chatterbox-TTS Worker (Ines-Stimme für Voxta)

# Basis: offizielles RunPod-Image mit CUDA 12.4
FROM runpod/base:0.4.4-cuda12.4.0

# --- Python 3.13 (chatterbox-tts braucht >=3.10; venv für saubere Deps) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.13 python3.13-venv python3.13-dev ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ENV VENV_PATH=/venv
RUN python3.13 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

WORKDIR /app

# --- Dependencies ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Handler + VoiceSamples (klein, im Image) ---
COPY handler.py .
COPY voices/ /models/voices/

# --- Environment ---
ENV MODEL_DIR=/models/chatterbox
ENV VOICES_DIR=/models/voices
ENV HF_DOWNLOAD=1
ENV DEVICE=cuda
ENV PRELOAD=1
ENV DEFAULT_VOICE=Voxta_F_Ines.wav

CMD ["python3.13", "-u", "handler.py"]
