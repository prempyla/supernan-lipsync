#!/usr/bin/env python3
"""
pipeline.py — Kannada → Hindi Dubbing Pipeline with Sync.so Lipsync
"""

import argparse
import base64
import logging
import os
import subprocess
import sys
import time
import shutil

import requests
from dotenv import load_dotenv

load_dotenv()


# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout)])
    return logging.getLogger("pipeline")

log = logging.getLogger("pipeline")


# ─── Config ───────────────────────────────────────────────────────────────────

CONFIG = {
    "video_fps": 25,
    "video_height": 720,
    "output_dir": "output",
    "sarvam_base_url": "https://api.sarvam.ai",
    "sync_api_url": "https://api.sync.so/v2/generate",
    "tts_speaker": "meera",
    "tts_model": "bulbul:v3",
    "translate_mode": "formal",
}


# ─── Errors ───────────────────────────────────────────────────────────────────

class PipelineError(Exception):
    pass

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _check_file(path: str, label: str):
    if not os.path.exists(path):
        raise PipelineError(f"{label} not found: {path}")
    if os.path.getsize(path) < 1:
        raise PipelineError(f"{label} is empty (0 bytes): {path}")
    log.debug(f"{label}: {path} ({os.path.getsize(path) / 1024:.0f} KB)")

def _run_ffmpeg(cmd: list, description: str):
    log.debug(f"ffmpeg: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = "\n".join(result.stderr.strip().split("\n")[-5:])
        raise PipelineError(f"{description} failed (exit {result.returncode}).\n{stderr}")
    return result

def _sarvam_post(endpoint: str, payload: dict, api_key: str, files: dict = None, description: str = "Sarvam API") -> dict:
    url = f"{CONFIG['sarvam_base_url']}/{endpoint}"
    headers = {"api-subscription-key": api_key}

    if files:
        response = requests.post(url, headers=headers, data=payload, files=files, timeout=60)
    else:
        headers["Content-Type"] = "application/json"
        response = requests.post(url, headers=headers, json=payload, timeout=60)

    if response.status_code == 401:
        raise PipelineError(f"{description}: Invalid API key.")
    if response.status_code == 402:
        raise PipelineError(f"{description}: Credits exhausted.")
    if response.status_code != 200:
        raise PipelineError(f"{description} failed (HTTP {response.status_code}): {response.text[:200]}")

    return response.json()

# ─── Stage 1: Extract ────────────────────────────────────────────────────────

def extract_clip(input_video: str, start: int, end: int, output_dir: str) -> dict:
    """Extract video clip and audio using ffmpeg."""
    log.info(f"{'='*60}")
    log.info(f"STAGE 1: EXTRACT ({start}s – {end}s, {end - start}s)")
    log.info(f"{'='*60}")

    _check_file(input_video, "Input video")

    duration = end - start
    if duration <= 0:
        raise PipelineError(f"Invalid time range: start={start}s >= end={end}s")

    os.makedirs(output_dir, exist_ok=True)
    fps = CONFIG["video_fps"]
    height = CONFIG["video_height"]

    paths = {
        "clip_video": os.path.join(output_dir, "clip_video.mp4"),
        "clip_audio": os.path.join(output_dir, "clip_audio.wav"),
    }

    _run_ffmpeg([
        "ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", input_video,
        "-vf", f"scale=-1:{height},fps={fps}", "-c:v", "libx264", "-preset", "fast", "-an",
        paths["clip_video"]
    ], "Video extraction")
    log.info(f"  ✓ Video: {paths['clip_video']}")

    _run_ffmpeg([
        "ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", input_video,
        "-vn", "-acodec", "pcm_f32le", "-ar", "16000", "-ac", "1",
        paths["clip_audio"]
    ], "Audio extraction")
    log.info(f"  ✓ Audio: {paths['clip_audio']}")

    return paths

# ─── Stage 2: Transcribe (Sarvam Saarika) ─────────────────────────────────────

def transcribe(audio_path: str, output_dir: str, api_key: str) -> str:
    """Transcribe Kannada audio → Kannada text using Sarvam Saarika STT."""
    log.info(f"{'='*60}")
    log.info(f"STAGE 2: TRANSCRIBE (Sarvam Saarika — Kannada)")
    log.info(f"{'='*60}")

    _check_file(audio_path, "Audio file")

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    response = _sarvam_post(
        "speech-to-text",
        payload={
            "language_code": "kn-IN",
            "model": "saarika:v2",
            "with_timestamps": False,
            "with_disfluencies": False,
        },
        files={"file": ("clip_audio.wav", audio_bytes, "audio/wav")},
        api_key=api_key,
        description="Saarika STT"
    )

    kannada_text = response.get("transcript", "").strip()
    if not kannada_text:
        raise PipelineError("Saarika returned empty transcript. Is the audio audible Kannada?")

    log.info(f"  ✓ Kannada: {kannada_text}")

    with open(os.path.join(output_dir, "kannada.txt"), "w", encoding="utf-8") as f:
        f.write(kannada_text)

    return kannada_text


# ─── Stage 3: Translate (Sarvam Mayura) ───────────────────────────────────────

def translate(kannada_text: str, output_dir: str, api_key: str) -> str:
    """Translate Kannada → Hindi using Sarvam Mayura (direct, no English hop)."""
    log.info(f"{'='*60}")
    log.info(f"STAGE 3: TRANSLATE (Sarvam Mayura — Kannada → Hindi)")
    log.info(f"{'='*60}")

    response = _sarvam_post(
        "translate",
        payload={
            "input": kannada_text,
            "source_language_code": "kn-IN",
            "target_language_code": "hi-IN",
            "speaker_gender": "Female",
            "mode": CONFIG["translate_mode"],
            "enable_preprocessing": True,
        },
        api_key=api_key,
        description="Mayura Translate"
    )

    hindi_text = response.get("translated_text", "").strip()
    if not hindi_text:
        raise PipelineError("Mayura returned empty translation. Check API credits.")

    # Validate Devanagari output
    devanagari = [c for c in hindi_text if '\u0900' <= c <= '\u097F']
    if not devanagari:
        log.warning("  ⚠ No Devanagari characters — output may be transliterated English")

    log.info(f"  ✓ Hindi: {hindi_text}")

    with open(os.path.join(output_dir, "hindi.txt"), "w", encoding="utf-8") as f:
        f.write(hindi_text)

    return hindi_text

# ─── Stage 4: TTS (Sarvam Bulbul v3) ─────────────────────────────────────────
def generate_speech(hindi_text: str, output_dir: str, api_key: str) -> str:
    """Generate Hindi speech using Sarvam Bulbul v3 TTS."""
    log.info(f"{'='*60}")
    log.info(f"STAGE 4: TTS (Sarvam Bulbul v3)")
    log.info(f"{'='*60}")
    if not hindi_text or not hindi_text.strip():
        raise PipelineError("Hindi text is empty — nothing to synthesise.")
    response = _sarvam_post(
        "text-to-speech",
        payload={
            "inputs": [hindi_text],
            "target_language_code": "hi-IN",
            "speaker": CONFIG["tts_speaker"],
            "model": CONFIG["tts_model"],
            "pitch": 0,
            "pace": 1.0,
            "loudness": 1.5,
            "enable_preprocessing": True,
        },
        api_key=api_key,
        description="Bulbul v3 TTS"
    )
    audio_b64 = response.get("audios", [None])[0]
    if not audio_b64:
        raise PipelineError("Bulbul v3 returned no audio. Check credits or text length.")
    wav_path = os.path.join(output_dir, "hindi_dubbed.wav")
    import base64
    with open(wav_path, "wb") as f:
        f.write(base64.b64decode(audio_b64))
    _check_file(wav_path, "Generated audio")
    log.info(f"  ✓ Audio: {wav_path} ({os.path.getsize(wav_path) / 1024:.0f} KB)")
    return wav_path
