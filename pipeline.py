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
