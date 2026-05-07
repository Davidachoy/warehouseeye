# WarehouseEye
## Operational Intelligence for Warehouse CCTV on AMD MI300X

![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![AMD MI300X](https://img.shields.io/badge/AMD-MI300X-red.svg)
![Qwen3-VL](https://img.shields.io/badge/VLM-Qwen3--VL--32B-blue.svg)
![lablab.ai](https://img.shields.io/badge/lablab.ai-hackathon-purple.svg)

## TL;DR
WarehouseEye turns long warehouse CCTV footage into searchable operational intelligence by extracting frames, tracking people persistently, and attaching structured semantic activity to each tracked identity.  
The pipeline combines `FrameExtractor`, `PersonDetector` (RT-DETRv2), `PersonTracker` (ByteTrack), `VisionAnalyzer` (Qwen3-VL via vLLM), and SQLite-backed query APIs in one workflow.  
AMD MI300X matters because its 192 GB HBM3 is large enough to co-run vision-language and ASR workloads on a single node without multi-GPU orchestration.

## The Problem
- Auditing warehouse CCTV manually is slow and expensive.
- Commercial solutions like BriefCam can cost around $50K/year.
- Generic VLMs (GPT-4V/Gemini-style APIs) often fail to preserve identity continuity across frames.

## The Insight
- ByteTrack keeps persistent `track_id` values per person over time (`warehouseeye/tracking/tracker.py`).
- Qwen3-VL-32B (Apache-2.0) adds crop-level activity JSON via `VisionAnalyzer` (`warehouseeye/gpu/vision_analyzer.py`).
- The pipeline is designed so detector, tracker, VLM, and Whisper stages can run in one AMD node (`warehouseeye/pipeline/orchestrator.py`, `warehouseeye/gpu/whisper_client.py`).

## Demo
- Hugging Face Space: [TODO]
- Streamlit app entrypoint: `frontend/app.py` (Space variant: `frontend/app_space.py`)
- Streamlit UI screenshot: `[TODO add screenshot file path, e.g. docs/space-hero.png]`

## Architecture
```text
Input Video
   |
   v
[1] Frame Extraction
    warehouseeye/ingestion/frame_extractor.py (FrameExtractor)
   |
   v
[2] Person Detection
    warehouseeye/tracking/detector.py (PersonDetector, RT-DETRv2)
   |
   v
[3] Multi-Object Tracking
    warehouseeye/tracking/tracker.py (PersonTracker, ByteTrack track_id)
   |
   v
[4] Crop Semantics
    warehouseeye/gpu/vision_analyzer.py (VisionAnalyzer, Qwen3-VL via VLLMClient)
   |
   v
[5] Persistence + Query
    warehouseeye/pipeline/db.py (SQLite tracks/identities/videos)
    api/main.py + api/query_resolver.py (NL query + timeline retrieval)
```

## Tech Stack
| Component | Library / Model | License |
|---|---|---|
| API backend | FastAPI (`api/main.py`) | MIT |
| UI | Streamlit (`frontend/app.py`) | Apache-2.0 |
| Frame extraction | PySceneDetect + OpenCV (`FrameExtractor`) | BSD-3-Clause + Apache-2.0 |
| Person detector | `PekingU/rtdetr_v2_r50vd` via Transformers (`PersonDetector`) | Apache-2.0 |
| Tracker | ByteTrack via `supervision` (`PersonTracker`) | MIT |
| Vision-language model | `Qwen/Qwen3-VL-32B-Instruct` via vLLM/OpenAI API format | Apache-2.0 |
| Speech model | `openai/whisper-large-v3-turbo` (`WhisperClient`) | MIT |
| Storage | SQLite (`warehouseeye.sqlite3`) | Public domain |

## Benchmarks (AMD MI300X, from `data/prerendered/*/benchmarks.json`)
| Video ID | Frames analyzed | Tokens/sec | Latency/crop (ms) | Cost/min video (USD) | WarehouseEye run cost (USD) | GPT-4V est. cost (USD)\* |
|---|---:|---:|---:|---:|---:|---:|
| `video` | 66 | 267.24 | 2313.47 | 0.0072 | 0.0130 | 0.0200 |
| `video2` | 4 | 354.56 | 1782.67 | 0.0146 | 0.0025 | 0.0038 |
| **Combined** | **70** | **272.44** | **2207.58** | **0.0078** | **0.0155** | **0.0238** |

\* GPT-4V estimate is derived from repo baseline logic (`vs_gpt4v_estimated_savings_pct = 35` in `api/pipeline_runner.py`) and pricing notes in `BENCHMARKS.md` ($10/M input, $30/M output).

## Re-ID extension
- WarehouseEye supports optional persistent Re-ID with two pluggable backends:
  - `Qwen/Qwen3-VL-Embedding-2B` (Apache-2.0) — multimodal foundation model embedding served via vLLM. Good for semantic matching.
  - `OSNet` via `torchreid` (MIT) — specialized person Re-ID model, much tighter same-person cosine clusters. Recommended default for clean identity continuity.
- License compatibility is preserved: only Apache-2.0 / MIT / BSD components are used.
- Choose backend with `REID_BACKEND=qwen|osnet` (default `qwen`).
- Positioning statement for the pitch: **First open-source video pipeline pairing a foundation-model semantic embedder with a specialized OSNet person Re-ID head, fully MIT/Apache-compatible.**

### Backend selection
```bash
# Specialized OSNet (MIT) - tighter same-person matching, runs locally on CPU/GPU
export REID_BACKEND=osnet
# Optional: override OSNet model variant
export OSNET_MODEL_NAME=osnet_x0_25     # 2 MB, default
# export OSNET_MODEL_NAME=osnet_x1_0    # heavier, higher accuracy
# export OSNET_MODEL_PATH=/abs/path/to/local/weights.pth  # offline weights

# Qwen3-VL multimodal embedding (Apache-2.0) - default
export REID_BACKEND=qwen
```

### 1) Launch both vLLM services (semantic + embedding)
```bash
scripts/launch_embedding_server.sh
```

This script launches:
- Semantic model on `http://localhost:8000/v1/models` with reduced GPU allocation (`--gpu-memory-utilization 0.6`)
- Embedding model on `http://localhost:8001/v1/models` with low GPU allocation (`--gpu-memory-utilization 0.05`)

It also applies AMD recipe environment settings:
- `MIOPEN_FIND_MODE=FAST`
- `VLLM_ROCM_USE_AITER=1`

### 2) Enable Re-ID in pipeline runs
Use either CLI flag or environment variable:
```bash
PYTHONPATH=. python scripts/test_pipeline_local.py --video-path "/absolute/path/to/video.mp4" --enable-reid
```

```bash
ENABLE_REID=1 PYTHONPATH=. python scripts/test_pipeline_local.py --video-path "/absolute/path/to/video.mp4"
```

Recommended defaults for indoor warehouse / kitchen scenes (validated to recover 1 ID for a 1-person clip and 2 IDs for a 2-person clip):

```bash
# Pipeline / detector
export WAREHOUSEEYE_SAMPLE_EVERY_SEC=0.5
export WAREHOUSEEYE_DETECTOR_THRESHOLD=0.40
export WAREHOUSEEYE_DETECTOR_INPUT_SIZE=1024
export WAREHOUSEEYE_MIN_BBOX_AREA=2500
export WAREHOUSEEYE_TRACKER_LOST_TRACK_BUFFER=30

# Re-ID matching
export WAREHOUSEEYE_REID_SIMILARITY_THRESHOLD=0.50
export WAREHOUSEEYE_REID_CROP_EXPAND_RATIO=-0.12
export WAREHOUSEEYE_REID_AGGREGATION=mean_topk
export WAREHOUSEEYE_REID_TOPK=3
export WAREHOUSEEYE_REID_TTA_HFLIP=1
export WAREHOUSEEYE_REID_MAX_ANCHORS=8
export WAREHOUSEEYE_REID_ANCHOR_MIN_DISTANCE=0.15
export WAREHOUSEEYE_REID_ANCHOR_MIN_SHARPNESS=0
export WAREHOUSEEYE_REID_MAX_LOST_AGE_SEC=600
export WAREHOUSEEYE_REID_ACTIVE_ANCHOR_REFRESH_EVERY=2
```

#### Re-ID quality knobs (introduced in this iteration)

| Env var | Default | What it does |
|---|---|---|
| `WAREHOUSEEYE_REID_AGGREGATION` | `max` | How to combine per-anchor similarities for a candidate track. `max` is brittle (one bad anchor poisons the score); `mean_topk` averages the best k anchors and is far more stable. |
| `WAREHOUSEEYE_REID_TOPK` | `3` | When `aggregation=mean_topk`, the number of top-anchor similarities to average. Falls back to mean of all when the candidate has fewer anchors. |
| `WAREHOUSEEYE_REID_TTA_HFLIP` | `0` | When `1`, also embed the horizontal-flip of the crop and average the two L2-normalized vectors. Doubles per-crop embedding latency; lifts same-person cosine ~5–15%. |
| `WAREHOUSEEYE_REID_MAX_ANCHORS` | `5` | Maximum pose-diverse anchors retained per track in the gallery. Higher = better recall on pose change, more memory. |
| `WAREHOUSEEYE_REID_ANCHOR_MIN_DISTANCE` | `0.15` | Novelty gate. New anchors are rejected when their cosine similarity to any existing anchor is `>= 1 - this`. |
| `WAREHOUSEEYE_REID_ANCHOR_MIN_SHARPNESS` | `0.0` | When `> 0`, gate anchor-promotion on Laplacian-variance focus measure. Rejects motion-blurred crops so they do not poison the gallery. Disabled by default. |
| `WAREHOUSEEYE_REID_MAX_LOST_AGE_SEC` | `300.0` | Window in which a `lost` track is eligible for Re-ID recovery. Bump for longer videos. |
| `WAREHOUSEEYE_REID_ACTIVE_ANCHOR_REFRESH_EVERY` | `0` | When `> 0`, every N frames recompute the embedding for an already-known active track and try to register it as an additional anchor (gated by novelty). This is what lets a track build a rich pose gallery during its initial active life so the *first* re-appearance after a gap can match. Disabled by default; set to `2`–`3` for short videos. |

Tuning workflow:

```bash
# 1) Run the pipeline with REID enabled to populate reid_attempts.
ENABLE_REID=1 PYTHONPATH=. python scripts/test_pipeline_local.py --video-path data/<video>.mp4 --base-dir data/<video> --enable-reid

# 2) See where same-person sims cluster vs near-miss rejections.
PYTHONPATH=. python scripts/dump_reid_distribution.py --db data/<video>/warehouseeye.sqlite3

# 3) Iterate fast on threshold/aggregation/TTA without re-running detection.
PYTHONPATH=. python scripts/reembed_from_crops.py --db data/<video>/warehouseeye.sqlite3 \
    --backend osnet --aggregation mean_topk --topk 3 --tta-hflip --threshold 0.50 --max-anchors 8

# 4) Get a data-driven threshold suggestion via IoU pseudo-labels.
PYTHONPATH=. python scripts/calibrate_reid_threshold.py --db data/<video>/warehouseeye.sqlite3
```

Every Re-ID decision (matched, below_threshold, no_candidates, anchor_rejected_blur) is logged to the `reid_attempts` table in the per-video SQLite, with `best_similarity`, `second_best_similarity`, candidate counts and anchor counts — the calibration scripts and the dashboards read from it.

### 3) Produce before/after Re-ID metrics
```bash
PYTHONPATH=. python scripts/test_reid_pipeline.py --video-path "data/warehouse_demo_1.mp4" \
    --dump-similarity-matrix data/reid_eval/similarity.csv
```

Outputs:
- `data/reid_comparison.json`
- Optional `--dump-similarity-matrix <path>`: CSV of every `reid_attempts` row for offline plotting.
- Terminal summary with:
  - ID switches (without Re-ID vs with Re-ID)
  - Unique tracks (without Re-ID vs with Re-ID)
  - Number of Re-ID matches
  - Average match similarity

Use these before/after numbers as your benchmark metric in demos and pitch materials.

## Setup
```bash
git clone https://github.com/[TODO-org]/warehouseeye.git
cd warehouseeye
pip install -r requirements.txt
PYTHONPATH=. python scripts/test_pipeline_local.py --video-path "/absolute/path/to/video.mp4"
```

## Why AMD MI300X
`WhisperClient` is implemented for `openai/whisper-large-v3-turbo`, and `VisionAnalyzer` drives Qwen3-VL through `VLLMClient`; this repo is structured to run those heavy multimodal stages in one orchestrated node (`Orchestrator._run_semantic_and_audio_parallel`).  
With 192 GB HBM3 on MI300X, you can target Qwen3-VL-32B + Whisper large-v3-turbo in parallel without sharding models across multiple GPUs, removing cross-device coordination overhead and simplifying deployment/ops.

## Acknowledgements
AMD, Hugging Face, the Qwen team, and lablab.ai.

## License
MIT (see `LICENSE`).