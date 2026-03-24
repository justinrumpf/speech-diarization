import requests
import base64
import json
import sys

POD_URL = "https://r74bo07kq50opn-8080.proxy.runpod.net"

# Read local test.mp3 and send as base64
with open("test.mp3", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

print(f"Sending test.mp3 ({len(audio_b64) // 1024} KB base64) to {POD_URL}/transcribe ...")

resp = requests.post(
    f"{POD_URL}/transcribe",
    json={
        "base64": audio_b64,
        "format": "mp3",
    },
    timeout=600,
)

print(f"Status: {resp.status_code}")
print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
