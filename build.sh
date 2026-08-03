#!/usr/bin/env bash
# Baut und pusht das RunPod-Worker-Image.
# Voraussetzungen: Docker installiert, bei Docker Hub angemeldet.
# Verwendung:  ./build.sh <dockerhub-user> [image-tag]
set -euo pipefail

USER="${1:?Usage: ./build.sh <dockerhub-user> [tag]}"
TAG="${2:-voxta-chatterbox-tts}"
IMAGE="$USER/$TAG:latest"

echo "=== Vorbereiten: Stimmen + Modell ins Build-Kontext kopieren ==="
# Von ki2000 (diesem Host):
mkdir -p build-context/voices build-context/chatterbox-model
cp -r /home/kitt/voxta/Data/Audio/VoiceSamples/. build-context/voices/ 2>/dev/null || true
cp -r /home/kitt/voxta/Data/Models/Chatterbox/chatterbox/. build-context/chatterbox-model/ 2>/dev/null || true

echo "Stimmen: $(ls build-context/voices | wc -l) Dateien"
echo "Modell: $(du -sh build-context/chatterbox-model | cut -f1)"

echo "=== Docker build ==="
docker build -t "$IMAGE" -f Dockerfile .

echo "=== Docker push ==="
docker push "$IMAGE"

echo "=== Aufräumen ==="
rm -rf build-context

echo ""
echo "FERTIG: $IMAGE"
echo "Im RunPod-Konsole Endpoint erstellen:"
echo "  - Custom Container: $IMAGE"
echo "  - GPU: RTX 4090 (oder A6000)"
echo "  - idle_timeout: 180"
echo "  - minWorkers: 0, maxWorkers: 1"
