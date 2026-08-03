#!/usr/bin/env python3
"""RunPod Serverless Handler: Chatterbox-TTS auf CUDA.

Erwartet im Job-Input:
{
  "text": "Hallo Michael, ich bin Ines.",
  "voice": "Voxta_F_Ines.wav",        // optional, Default siehe DEFAULT_VOICE
  "exaggeration": 0.5,                 // optional
  "repetition_penalty": 2.0,           // optional
  "cfg_weight": 0.5,                   // optional
  "temperature": 0.8                   // optional
}

Liefert im Job-Output:
{
  "audio_b64": "<base64-WAV>",
  "sample_rate": 24000,
  "duration_s": 4.8,
  "seconds_on_gpu": 2.1
}
"""
import base64
import io
import os
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
import wave
import runpod

from chatterbox.tts import ChatterboxTTS

# Modell-Verzeichnis: wird beim Build ins Image gelegt ODER beim ersten
# Cold Start von HuggingFace gezogen (dann HF_DOWNLOAD=True).
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/chatterbox")
HF_DOWNLOAD = os.environ.get("HF_DOWNLOAD", "0") == "1"

# Stimmen-Verzeichnis: enthält die Voxta_F_*.wav + .ref_text.txt Dateien.
VOICES_DIR = os.environ.get("VOICES_DIR", "/models/voices")

DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "Voxta_F_Ines.wav")
DEVICE = os.environ.get("DEVICE", "cuda")

_tts = None
_voices = []


def load_model():
    """Lädt Chatterbox einmalig (Cold Start), cached global."""
    global _tts, _voices
    if _tts is not None:
        return _tts

    t0 = time.time()
    ckpt_dir = MODEL_DIR
    if HF_DOWNLOAD:
        from huggingface_hub import snapshot_download
        ckpt_dir = snapshot_download(repo_id="ResembleAI/chatterbox")
        print(f"HF-Download abgeschlossen: {ckpt_dir}", flush=True)

    if not os.path.isdir(ckpt_dir):
        raise RuntimeError(f"Modell-Verzeichnis fehlt: {ckpt_dir}")

    _tts = ChatterboxTTS.from_local(ckpt_dir, DEVICE)
    print(f"Modell geladen in {time.time()-t0:.1f}s auf {DEVICE}", flush=True)

    # Stimmen-Liste aus VOICES_DIR
    if os.path.isdir(VOICES_DIR):
        _voices = sorted(f for f in os.listdir(VOICES_DIR) if f.endswith(".wav"))
    print(f"Verfügbare Stimmen ({len(_voices)}): {_voices[:5]}...", flush=True)
    return _tts


def synthesize(text, voice, exaggeration, repetition_penalty, cfg_weight, temperature):
    """Einzel-Synthese mit Vorbereitung der Referenzstimme."""
    tts = load_model()

    # Stimme vorbereiten (jeder Request, weil Serverless stateless Sessions hat)
    t1 = time.time()
    voice_path = os.path.join(VOICES_DIR, voice) if voice else None
    if voice_path and os.path.isfile(voice_path):
        tts.prepare_conditionals(voice_path, exaggeration=exaggeration)
    else:
        # Fallback: Default-Stimme
        default_path = os.path.join(VOICES_DIR, DEFAULT_VOICE)
        if os.path.isfile(default_path):
            tts.prepare_conditionals(default_path, exaggeration=exaggeration)

    t2 = time.time()
    wav = tts.generate(
        text,
        repetition_penalty=repetition_penalty,
        cfg_weight=cfg_weight,
        temperature=temperature,
    )
    t3 = time.time()

    # Tensor → float32 numpy → int16 PCM
    if isinstance(wav, torch.Tensor):
        wav = wav.float().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32)
    pcm = (wav * 32767).astype(np.int16)

    # WAV in Bytes (stdlib wave, libsndfile kann auf manchen Hosts defekt sein)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(tts.sr)
        wf.writeframes(pcm.tobytes())
    wav_bytes = buf.getvalue()

    return {
        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
        "sample_rate": tts.sr,
        "duration_s": round(len(pcm) / tts.sr, 2),
        "prepare_s": round(t2 - t1, 2),
        "generate_s": round(t3 - t2, 2),
    }


def handler(job):
    """RunPod Job-Handler."""
    job_input = job.get("input", {})
    text = job_input.get("text", "")
    if not text:
        return {"error": "Kein 'text' im Input"}

    voice = job_input.get("voice", DEFAULT_VOICE)
    exaggeration = float(job_input.get("exaggeration", 0.5))
    repetition_penalty = float(job_input.get("repetition_penalty", 2.0))
    cfg_weight = float(job_input.get("cfg_weight", 0.5))
    temperature = float(job_input.get("temperature", 0.8))

    # TTS ist CPU-seitig kaum belastend, aber CUDA-GPU-Zeit messen
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    t_start = time.time()
    result = synthesize(text, voice, exaggeration, repetition_penalty, cfg_weight, temperature)
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    result["seconds_on_gpu"] = round(time.time() - t_start, 2)
    return result


# Modell direkt beim Start laden (Cold Start wird so in die Container-Initialisierung gelegt,
# NICHT in die erste Request-Antwort — wichtig, weil RunPod die Init-Zeit auch abrechnet,
# aber der erste Request dann schnell antwortet).
if os.environ.get("PRELOAD", "1") == "1":
    load_model()

runpod.serverless.start({"handler": handler})
