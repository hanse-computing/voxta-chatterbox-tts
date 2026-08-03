# RunPod Serverless: Chatterbox-TTS Worker (Ines-Stimme für Voxta)

# Basis: offizielles RunPod-Image mit CUDA 12.9
FROM runpod/base:1.1.0-cuda1290-ubuntu2204

# --- Python (chatterbox-tts braucht >=3.10) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev python3-pip ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ENV VENV_PATH=/venv
RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

WORKDIR /app

# --- Dependencies ---
COPY requirements.txt .
# Python-Version prüfen (chatterbox-tts braucht >=3.10)
RUN python3 --version && pip install --no-cache-dir -r requirements.txt

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

CMD ["python3", "-u", "handler.py"]
