import requests
import base64
import json

POD_URL = "https://4xcjax634z1gsq-8080.proxy.runpod.net"

# --- Test 1: Single request with ID ---
print("=== Test 1: Single transcribe with ID ===")
resp = requests.post(
    f"{POD_URL}/transcribe",
    json={
        "id": "single-test-001",
        "url": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav",
    },
    timeout=600,
)
print(f"Status: {resp.status_code}")
try:
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
except Exception:
    print(resp.text)

# --- Test 2: Batch request ---
print("\n" + "=" * 60)
print("=== Test 2: Batch transcribe ===")

with open("test.mp3", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

resp2 = requests.post(
    f"{POD_URL}/transcribe/batch",
    json={
        "items": [
            {
                "id": "batch-url-001",
                "url": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav",
            },
            {
                "id": "batch-b64-002",
                "base64": audio_b64,
                "format": "mp3",
            },
        ]
    },
    timeout=1200,
)
print(f"Status: {resp2.status_code}")
try:
    print(json.dumps(resp2.json(), indent=2, ensure_ascii=False))
except Exception:
    print(resp2.text)
