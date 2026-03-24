#!/bin/bash
set -e

# Deploy script for RunPod pod: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

echo "=== Speech Diarization Deploy ==="

# System dependencies
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

# 1. Pin torch to match RunPod base image (CUDA 12.4) — must be first
pip install --no-cache-dir torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# 2. Install remaining Python dependencies
pip install --no-cache-dir -r requirements.txt --ignore-installed blinker

# 3. Pin huggingface_hub for pyannote 3.x compatibility (use_auth_token support)
pip install --no-cache-dir "huggingface_hub<0.24.0"

# 4. Restore cuDNN version matching CUDA 12.4 (pip may overwrite it)
pip install --no-cache-dir nvidia-cudnn-cu12==9.1.0.70

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "WARNING: HF_TOKEN not set. Diarization will fail."
    echo "Run: export HF_TOKEN=your_token"
    echo ""
fi

echo "=== Deploy complete ==="
echo "Run: cd $REPO_DIR && python3 app.py"
