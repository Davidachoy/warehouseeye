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