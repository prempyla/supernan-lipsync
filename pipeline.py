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
    "tts_speaker": "anushka",
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

def _sarvam_post(endpoint: str, payload: dict, api_key: str,
                 files: dict = None, description: str = "Sarvam API") -> dict:
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

def _get_duration(path: str) -> float:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True
    )
    return float(res.stdout.strip())

def _upload_file_public(file_path: str) -> str:
    """Upload file to uguu.se and return a public URL (24hr expiry)."""
    with open(file_path, "rb") as f:
        resp = requests.post(
            "https://uguu.se/upload",
            files={"files[]": (os.path.basename(file_path), f)}
        )
    if resp.status_code != 200:
        raise PipelineError(f"File upload failed (HTTP {resp.status_code}): {resp.text[:200]}")
    data = resp.json()
    url = data["files"][0]["url"]
    log.info(f"  Public URL: {url}")
    return url


# ─── Stage 1: Extract ────────────────────────────────────────────────────────

def extract_clip(input_video: str, start: int, end: int, output_dir: str) -> dict:
    """Extract video clip and audio using ffmpeg."""
    log.info(f"{'='*60}")
    log.info(f"STAGE 1: EXTRACT ({start}s – {end}s, {end - start}s)")
    log.info(f"{'='*60}")

    _check_file(input_video, "Input video")

    if end <= start:
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


# ─── Stage 2: Transcribe (Sarvam Saarika) ────────────────────────────────────

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


# ─── Stage 3: Translate (Sarvam Mayura) ──────────────────────────────────────

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
    with open(wav_path, "wb") as f:
        f.write(base64.b64decode(audio_b64))

    _check_file(wav_path, "Generated audio")
    log.info(f"  ✓ Audio: {wav_path} ({os.path.getsize(wav_path) / 1024:.0f} KB)")
    return wav_path


# ─── Stage 5: Audio Sync ─────────────────────────────────────────────────────

def sync_audio(dubbed_path: str, clip_audio_path: str, output_dir: str) -> str:
    """Match dubbed audio duration to video duration using ffmpeg atempo."""
    log.info(f"{'='*60}")
    log.info(f"STAGE 5: AUDIO SYNC")
    log.info(f"{'='*60}")

    _check_file(dubbed_path, "Dubbed audio")
    _check_file(clip_audio_path, "Original audio")

    video_dur = _get_duration(clip_audio_path)
    dub_dur = _get_duration(dubbed_path)
    ratio = dub_dur / video_dur

    log.info(f"  Video: {video_dur:.2f}s | Hindi TTS: {dub_dur:.2f}s | ratio: {ratio:.3f}x")

    synced_path = os.path.join(output_dir, "hindi_synced.wav")

    if abs(ratio - 1.0) < 0.02:
        shutil.copy(dubbed_path, synced_path)
        log.info(f"  ✓ Already within 2% — no adjustment needed")
    else:
        if 0.5 <= ratio <= 2.0:
            atempo = f"atempo={ratio:.4f}"
        elif ratio > 2.0:
            atempo = f"atempo=2.0,atempo={ratio / 2.0:.4f}"
        else:
            atempo = f"atempo=0.5,atempo={ratio / 0.5:.4f}"

        _run_ffmpeg([
            "ffmpeg", "-y", "-i", dubbed_path,
            "-filter:a", atempo, synced_path
        ], "Tempo adjustment")
        log.info(f"  ✓ Synced: {atempo}")

    return synced_path


# ─── Stage 6: Lipsync (sync.so) ──────────────────────────────────────────────

def lipsync(video_path: str, audio_path: str, output_dir: str, api_key: str) -> str:
    """Lip-sync video to audio using sync.so API."""
    log.info(f"{'='*60}")
    log.info(f"STAGE 6: LIPSYNC (sync.so)")
    log.info(f"{'='*60}")

    _check_file(video_path, "Video clip")
    _check_file(audio_path, "Synced audio")

    headers = {"x-api-key": api_key}

    # Convert WAV to MP3 (sync.so needs encoded audio, not raw PCM)
    mp3_path = audio_path
    if audio_path.endswith(".wav"):
        mp3_path = audio_path.replace(".wav", ".mp3")
        _run_ffmpeg([
            "ffmpeg", "-y", "-i", audio_path,
            "-codec:a", "libmp3lame", "-qscale:a", "2", mp3_path
        ], "WAV to MP3 conversion")
        log.info(f"  ✓ Converted to MP3: {mp3_path}")

    # Upload files to get public URLs
    log.info("  Uploading video...")
    video_url = _upload_file_public(video_path)

    log.info("  Uploading audio...")
    audio_url = _upload_file_public(mp3_path)

    # Submit lipsync job
    log.info("  Submitting lipsync job...")
    job_resp = requests.post(
        CONFIG["sync_api_url"],
        headers={**headers, "Content-Type": "application/json"},
        json={
            "model": "lipsync-2",
            "input": [
                {"type": "video", "url": video_url},
                {"type": "audio", "url": audio_url},
            ]
        },
        timeout=30
    ).json()

    job_id = job_resp.get("id")
    if not job_id:
        raise PipelineError(f"sync.so job submission failed: {job_resp}")
    log.info(f"  ✓ Job submitted (id: {job_id}). Polling...")

    # Poll for completion (up to 5 min)
    out_url = None
    for attempt in range(60):
        time.sleep(5)
        status_resp = requests.get(
            f"https://api.sync.so/v2/generate/{job_id}",
            headers=headers, timeout=30
        ).json()
        status = status_resp.get("status", "unknown")
        log.debug(f"  Poll {attempt + 1}: {status}")

        if status.upper() == "COMPLETED":
            out_url = status_resp.get("outputUrl")
            break
        if status.upper() in ("FAILED", "ERROR", "REJECTED"):
            raise PipelineError(f"sync.so job {status}: {status_resp.get('error', 'unknown')}")

    if not out_url:
        raise PipelineError(f"sync.so timed out after 5 min. Job: {job_id}")

    # Download result
    output_path = os.path.join(output_dir, "final_dubbed.mp4")
    log.info("  Downloading result...")
    dl = requests.get(out_url, timeout=120)
    with open(output_path, "wb") as f:
        f.write(dl.content)

    _check_file(output_path, "Lip-synced video")
    log.info(f"  ✓ Done: {output_path} ({os.path.getsize(output_path) / (1024*1024):.1f} MB)")
    return output_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Kannada → Hindi Dubbing Pipeline with sync.so Lipsync"
    )
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--start", type=int, default=0, help="Clip start (seconds)")
    parser.add_argument("--end", type=int, required=True, help="Clip end (seconds)")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--sarvam-key", default=None,
                        help="Sarvam AI API key (or set SARVAM_API_KEY in .env)")
    parser.add_argument("--sync-key", default=None,
                        help="sync.so API key (or set SYNC_API_KEY in .env)")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")

    args = parser.parse_args()

    global log
    log = setup_logging(verbose=args.verbose)

    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    sarvam_key = args.sarvam_key or os.getenv("SARVAM_API_KEY")
    sync_key = args.sync_key or os.getenv("SYNC_API_KEY")

    if not sarvam_key:
        raise PipelineError("Sarvam API key required. Set SARVAM_API_KEY in .env or use --sarvam-key")
    if not sync_key:
        raise PipelineError("sync.so API key required. Set SYNC_API_KEY in .env or use --sync-key")

    log.info("")
    log.info("█" * 60)
    log.info("  KANNADA → HINDI DUBBING PIPELINE (sync.so)")
    log.info("█" * 60)
    log.info(f"  Input   : {args.input}")
    log.info(f"  Segment : {args.start}s – {args.end}s ({args.end - args.start}s)")
    log.info("█" * 60)

    start_time = time.time()

    try:
        # Stage 1: Extract
        paths = extract_clip(args.input, args.start, args.end, output_dir)

        # Stage 2: Transcribe
        kannada_text = transcribe(paths["clip_audio"], output_dir, sarvam_key)

        # Stage 3: Translate
        hindi_text = translate(kannada_text, output_dir, sarvam_key)

        # Stage 4: TTS
        dubbed_path = generate_speech(hindi_text, output_dir, sarvam_key)

        # Stage 5: Audio Sync
        synced_path = sync_audio(dubbed_path, paths["clip_audio"], output_dir)

        # Stage 6: Lipsync
        final_path = lipsync(paths["clip_video"], synced_path, output_dir, sync_key)

        if args.output:
            shutil.copy(final_path, args.output)
            final_path = args.output

        total = time.time() - start_time
        log.info("")
        log.info("█" * 60)
        log.info(f"  ✅ PIPELINE COMPLETE")
        log.info(f"  Output: {final_path}")
        log.info(f"  Total : {total:.1f}s")
        log.info("█" * 60)

    except PipelineError as e:
        log.error(f"\n❌ Pipeline failed:\n{e}")
        sys.exit(1)
    except KeyboardInterrupt:
        log.warning("\n⚠ Interrupted.")
        sys.exit(130)

if __name__ == "__main__":
    main()
