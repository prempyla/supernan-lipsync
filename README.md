# Supernan Lipsync Pipeline

Kannada → Hindi dubbing pipeline with AI lip-sync using [sync.so](https://sync.so).

## Pipeline

```
Kannada Video
    │
    ├─ [1] ffmpeg           → Extract clip + audio
    ├─ [2] Sarvam Saarika   → Kannada audio → Kannada text
    ├─ [3] Sarvam Mayura    → Kannada text → Hindi text
    ├─ [4] Sarvam Bulbul v3 → Hindi text → Hindi audio
    ├─ [5] ffmpeg atempo    → Match audio duration to video
    └─ [6] sync.so          → Lip-sync video to Hindi audio
    │
    ▼
  final_dubbed.mp4
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
```

## Usage

```bash
python pipeline.py --input video.mp4 --start 3 --end 18
```

## API Keys Required

| Service | Purpose | Get it at |
|---|---|---|
| Sarvam AI | STT + Translation + TTS | [dashboard.sarvam.ai](https://dashboard.sarvam.ai) |
| sync.so | Lipsync | [sync.so](https://sync.so) |

## Dependencies

- Python 3.8+
- ffmpeg (must be installed and in PATH)
- No GPU required — fully API-based
