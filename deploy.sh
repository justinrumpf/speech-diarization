#!/bin/bash
set -e

echo "=== Speech Diarization Deploy ==="

# Install system dependencies
apt-get update && apt-get install -y --no-install-recommends ffmpeg

# Clone or pull repo
REPO_DIR="/workspace/speech-diarization"
if [ -d "$REPO_DIR" ]; then
    echo "Repo exists, pulling latest..."
    cd "$REPO_DIR"
    git pull origin main
else
    echo "Cloning repo..."
    git clone https://github.com/justinrumpf/speech-diarization.git "$REPO_DIR"
    cd "$REPO_DIR"
fi

# Pin torch to match RunPod base image (CUDA 12.4)
pip install --no-cache-dir torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# Pin pyannote.audio to 3.x (4.x requires torch>=2.8)
pip install --no-cache-dir "pyannote.audio>=3.3.0,<4.0.0"

# Install remaining Python dependencies
pip install --no-cache-dir -r requirements.txt --ignore-installed blinker

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. Diarization will fail."
    echo "Run: export HF_TOKEN=your_token"
fi

echo "=== Deploy complete ==="
echo "Start the server with: cd $REPO_DIR && python3 app.py"
