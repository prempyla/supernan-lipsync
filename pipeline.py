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
