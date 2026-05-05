# WarehouseEye

Open-source operational intelligence pipeline for fixed-camera warehouse CCTV: scene-based frame extraction, RT-DETRv2 person detection, ByteTrack tracking, dominant clothing color tags, and SQLite storage.

## Requirements

- Python 3.10+ (3.14 used in development)
- [ffmpeg](https://ffmpeg.org/) on your `PATH` (for audio extraction)
- Optional: CUDA for faster inference

## Setup

```bash
cd warehouseeye
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Tests

From the repository root:

```bash
PYTHONPATH=. pytest tests/
```

## Local pipeline

Provide a path to your warehouse video (not included in the repo). Put copies under `data/videos/` if you like; that folder is gitignored.

```bash
PYTHONPATH=. python scripts/test_pipeline_local.py /path/to/video.mp4
# or equivalently:
PYTHONPATH=. python scripts/test_pipeline_local.py --video-path "/path/with spaces/video.mp4"
```

Optional: `--min-tracks N` checks that at least `N` distinct identities were written (default `1`). Use `--min-tracks 3` only when your clip reliably shows several people across the sampled frames.

Artifacts are written under `data/` (videos, frames, crops, audio, SQLite). Those output paths are listed in `.gitignore`; `data/.gitkeep` keeps the folder in the repo without committing large media.

## License

See [LICENSE](LICENSE).