import os
import tempfile
import logging
import base64
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch
import av
import requests
from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline as DiarizationPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Configuration via environment variables ---
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
DEVICE = os.environ.get("DEVICE", "cuda")  # "cuda" or "cpu"
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16")  # "float16", "int8", "int8_float16"
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Required for pyannote diarization models

# --- Lazy-loaded globals ---
_whisper_model = None
_diarization_pipeline = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        logger.info("Loading Whisper model: %s (device=%s, compute=%s)", WHISPER_MODEL, DEVICE, COMPUTE_TYPE)
        _whisper_model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _whisper_model


def get_diarization_pipeline():
    global _diarization_pipeline
    if _diarization_pipeline is None:
        if not HF_TOKEN:
            raise RuntimeError(
                "HF_TOKEN environment variable is required for pyannote diarization. "
                "Get a token at https://huggingface.co/settings/tokens and accept the "
                "model terms at https://huggingface.co/pyannote/speaker-diarization-3.1"
            )
        logger.info("Loading pyannote diarization pipeline")
        _diarization_pipeline = DiarizationPipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN,
        )
        if DEVICE == "cuda":
            _diarization_pipeline.to(torch.device("cuda"))
    return _diarization_pipeline


def load_audio(path: str):
    """Load audio file using PyAV (bundled with faster-whisper). Returns (waveform, sample_rate)."""
    container = av.open(path)
    stream = container.streams.audio[0]
    sample_rate = stream.rate or stream.codec_context.sample_rate
    frames = []
    for frame in container.decode(audio=0):
        frames.append(frame.to_ndarray())
    container.close()
    audio = np.concatenate(frames, axis=1)  # (channels, samples)
    waveform = torch.from_numpy(audio).float()
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # mix to mono
    return waveform, sample_rate


def download_audio(url: str) -> str:
    """Download audio from a URL to a temporary file. Returns the temp file path."""
    resp = requests.get(url, timeout=300, stream=True)
    resp.raise_for_status()
    suffix = Path(url.split("?")[0]).suffix or ".mp3"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    for chunk in resp.iter_content(chunk_size=8192):
        tmp.write(chunk)
    tmp.close()
    return tmp.name


def resolve_audio_path(data: dict) -> str:
    """Resolve the audio source from the request. Returns a local file path and whether it's temporary."""
    if "url" in data:
        return download_audio(data["url"]), True
    elif "base64" in data:
        suffix = data.get("format", ".mp3")
        if not suffix.startswith("."):
            suffix = "." + suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(base64.b64decode(data["base64"]))
        tmp.close()
        return tmp.name, True
    elif "file_path" in data:
        path = data["file_path"]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        return path, False
    else:
        raise ValueError("Request must include 'url', 'base64', or 'file_path'")


def assign_speakers_to_segments(transcript_segments, diarization):
    """
    Match each transcribed word to a speaker using diarization timestamps.
    Returns a list of conversation turns with speaker labels.
    """
    # Build a flat list of words with timestamps
    words = []
    for seg in transcript_segments:
        if seg.words:
            for w in seg.words:
                words.append({
                    "start": w.start,
                    "end": w.end,
                    "word": w.word.strip(),
                })

    if not words:
        # Fallback: no word-level timestamps, use segment-level
        return _assign_speakers_to_segments_no_words(transcript_segments, diarization)

    # For each word, find the speaker from diarization who overlaps the most
    for word in words:
        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            overlap_start = max(word["start"], turn.start)
            overlap_end = min(word["end"], turn.end)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        word["speaker"] = best_speaker

    # Group consecutive words by speaker into turns
    turns = []
    current_speaker = None
    current_words = []
    current_start = 0.0

    for word in words:
        if word["speaker"] != current_speaker:
            if current_words:
                turns.append({
                    "speaker": current_speaker,
                    "start": round(current_start, 3),
                    "end": round(current_words[-1]["end"], 3),
                    "text": " ".join(w["word"] for w in current_words),
                })
            current_speaker = word["speaker"]
            current_words = [word]
            current_start = word["start"]
        else:
            current_words.append(word)

    if current_words:
        turns.append({
            "speaker": current_speaker,
            "start": round(current_start, 3),
            "end": round(current_words[-1]["end"], 3),
            "text": " ".join(w["word"] for w in current_words),
        })

    return turns


def _assign_speakers_to_segments_no_words(transcript_segments, diarization):
    """Fallback: assign speakers at the segment level when word timestamps aren't available."""
    turns = []
    for seg in transcript_segments:
        mid = (seg.start + seg.end) / 2
        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            overlap_start = max(seg.start, turn.start)
            overlap_end = min(seg.end, turn.end)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        turns.append({
            "speaker": best_speaker,
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
        })
    return turns


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    POST JSON with either:
      {"url": "https://example.com/audio.mp3"}
    or:
      {"file_path": "/path/to/local/file.mp3"}

    Optional params:
      - num_speakers: int (hint for diarization)
      - min_speakers: int
      - max_speakers: int
      - language: str (e.g. "en", auto-detected if omitted)
      - beam_size: int (default 5)

    Returns JSON with diarized transcript.
    """
    data = request.get_json(force=True)
    temp_file = None

    try:
        audio_path, is_temp = resolve_audio_path(data)
        if is_temp:
            temp_file = audio_path

        # --- Transcribe with faster-whisper ---
        model = get_whisper_model()
        segments_gen, info = model.transcribe(
            audio_path,
            beam_size=data.get("beam_size", 5),
            word_timestamps=True,
            vad_filter=True,
            language=data.get("language"),
        )
        # Materialize segments (triggers actual transcription)
        segments = list(segments_gen)

        logger.info(
            "Transcription complete: language=%s (prob=%.2f), duration=%.1fs",
            info.language, info.language_probability, info.duration,
        )

        # --- Diarize with pyannote ---
        diarization_pipeline = get_diarization_pipeline()
        diarize_kwargs = {}
        if "num_speakers" in data:
            diarize_kwargs["num_speakers"] = data["num_speakers"]
        if "min_speakers" in data:
            diarize_kwargs["min_speakers"] = data["min_speakers"]
        if "max_speakers" in data:
            diarize_kwargs["max_speakers"] = data["max_speakers"]

        waveform, sample_rate = load_audio(audio_path)
        diarize_output = diarization_pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            **diarize_kwargs,
        )
        # Newer pyannote wraps result in DiarizeOutput; extract the annotation
        if hasattr(diarize_output, "speaker_diarization"):
            diarization = diarize_output.speaker_diarization
        else:
            diarization = diarize_output

        # --- Merge transcription + diarization ---
        turns = assign_speakers_to_segments(segments, diarization)

        return jsonify({
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration": round(info.duration, 3),
            "num_speakers": len({t["speaker"] for t in turns}),
            "turns": turns,
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.exception("Transcription failed")
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)