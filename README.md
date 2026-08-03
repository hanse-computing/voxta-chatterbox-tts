# RunPod Serverless: Chatterbox-TTS Worker (Ines-Stimme für Voxta)

Deployt Chatterbox-TTS auf RunPod Serverless. Das Modell wird beim ersten
Cold Start von HuggingFace geladen (`ResembleAI/chatterbox`, ~13,9 GB) — so
bleibt das Repo klein und der Build schnell.

## Struktur
- `handler.py` — RunPod Serverless Handler (Chatterbox-Synthese, CUDA)
- `Dockerfile` — Basis: `runpod/base:0.4.4-cuda12.4.0`, Python 3.13 venv
- `requirements.txt` — chatterbox-tts 0.1.7 + torch 2.9.1 + runpod
- `voices/` — Voxta VoiceSamples (inkl. `Voxta_F_Ines.wav`)
- `builder.py` — optionales Testskript (lokal, CPU)

## Environment (am Endpoint setzen)
- `MODEL_DIR=/models/chatterbox` (Default, wird von HF gefüllt)
- `HF_DOWNLOAD=1` (Default: Modell von HuggingFace laden)
- `VOICES_DIR=/models/voices` (Default: im Image enthalten)
- `DEFAULT_VOICE=Voxta_F_Ines.wav`
- `DEVICE=cuda`
- `PRELOAD=1` (Modell beim Container-Start laden → erster Request schnell)

## Deploy (RunPod Console → Serverless → New Endpoint)
1. Quelle: GitHub-Repo (dieses)
2. GPU: RTX 4090 (oder A6000)
3. Idle Timeout: 180 s
4. Max Workers: 1, Min Workers: 0
5. FlashBoot: an (falls verfügbar)
