# WarehouseEye Final Benchmark Report

This file is intended to be filled after the final 32B run completes.

## Run metadata

- Date:
- Commit:
- Instance type:
- Region:
- Hourly instance cost (USD): `1.99`

## Metrics per video

| Video ID | Duration (sec) | Tracks | VLM requests | Total tokens | Avg latency/crop (ms) | Wall time (sec) | Throughput (tokens/sec) | Estimated run cost (USD) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `warehouse_demo_1` |  |  |  |  |  |  |  |  |
| `warehouse_demo_2` |  |  |  |  |  |  |  |  |

## 8B vs 32B quality and speed

| Dimension | Qwen3-VL-8B | Qwen3-VL-32B | Notes |
|---|---|---|---|
| Detail level |  |  |  |
| Accuracy / grounding |  |  |  |
| Safety relevance |  |  |  |
| Mean latency per crop |  |  |  |
| Total wall time (same sample) |  |  |  |

Evidence artifacts:

- HTML comparison report: `comparing_8b_vs_32b.html`
- Raw comparison payload: `comparing_8b_vs_32b.json`

## Milestone cost summary

| Cost item | Formula | Value (USD) |
|---|---|---:|
| Compute runtime | `instance_hours * 1.99` |  |
| Storage / transfer | `archive + upload` |  |
| Total milestone cost | `compute + storage` |  |

## GPT-4 Vision API counterfactual

Reference pricing used by repository helpers:

- Input: `$10 / 1M tokens`
- Output: `$30 / 1M tokens`

| Scenario | Input tokens | Output tokens | Estimated GPT-4V cost (USD) | WarehouseEye run cost (USD) | Savings (USD) | Savings (%) |
|---|---:|---:|---:|---:|---:|---:|
| `warehouse_demo_1` |  |  |  |  |  |  |
| `warehouse_demo_2` |  |  |  |  |  |  |
| Total |  |  |  |  |  |  |

Notes:

- Use `benchmarks.json` generated per video under `data/prerendered/<video_id>/`.
- If exact usage is unavailable, estimate tokens conservatively as `request_count * max_tokens`.

## Hugging Face Space prerender checklist

For each video in `data/prerendered_space/<video_id>/`, verify the following:

- `timeline.json`
- `tracks.db`
- `identities.json`
- `benchmarks.json`
- `crops/` (resized representative crops)
- `thumbnails/` (PNG keyframes with bounding boxes)

Root manifest:

- `data/prerendered_space/index.json`

If the Space runtime still expects `data/<video_id>/timeline.json`, either mount `data/prerendered_space` as the data root or symlink each `data/<video_id>` to `data/prerendered_space/<video_id>`.
