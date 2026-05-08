---
title: WarehouseEye
emoji: 📦
colorFrom: orange
colorTo: blue
sdk: streamlit
sdk_version: "1.41.0"
app_file: frontend/app_space.py
pinned: true
license: mit
short_description: Industrial video intelligence on AMD MI300X
tags:
  - amd-hackathon
  - amd-mi300x
  - qwen3-vl
  - vision-language
  - object-tracking
  - warehouse-analytics
  - on-premise
  - open-source
---

# WarehouseEye

Operational intelligence for warehouse CCTV, built as an open-source pipeline and benchmarked on AMD MI300X.

This Space is a **CPU-friendly pre-rendered demo** designed for public exploration. It shows what the system can do while keeping the runtime lightweight for free-tier hosting.

## Why this demo exists

Warehouse teams often need to audit long CCTV footage to answer operational and safety questions. Manual review is expensive, slow, and inconsistent.

WarehouseEye converts raw footage into structured timelines with:

- person tracking
- crop-level vision-language interpretation
- searchable event summaries
- benchmark and cost reporting

This Space exposes those results through a Streamlit UI with three tabs:

- **Operation Overview**
- **Ask the Video**
- **Performance Dashboard**

## What runs here vs full system

This Space intentionally runs in a constrained mode:

- no FastAPI backend calls
- no live vLLM inference
- no model hosting in the Space

Instead, it reads pre-rendered outputs from `data/prerendered/<video_id>/`:

- `timeline.json`
- `benchmarks.json`
- `videos/input_video.mp4`
- selected track crops

For free-form natural-language queries and live inference, deploy the full repository locally or on AMD GPU infrastructure.

## AMD MI300X relevance

WarehouseEye is tuned to showcase production-minded multimodal workloads on AMD hardware. MI300X provides 192 GB HBM, which is especially useful for single-node deployment of larger vision-language systems without coordinating multi-GPU sharding.

The performance dashboard in this Space reports **real measured benchmark outputs** captured from MI300X runs and stored in the pre-rendered artifacts.

## Screenshots

Add your images under `docs/` and keep these links:

![WarehouseEye hero section](docs/space-hero.png)
![Ask the Video tab](docs/space-query.png)
![Performance dashboard](docs/space-performance.png)

## Links

- GitHub: [https://github.com/Davidachoy/warehouseeye](https://github.com/Davidachoy/warehouseeye)
- Hugging Face Space: [https://huggingface.co/spaces/<your-hf-org>/warehouseeye](https://huggingface.co/spaces/<your-hf-org>/warehouseeye)
- Author X/Twitter: [https://x.com/achoy__](https://x.com/achoy__)

## Like this Space

If WarehouseEye is useful or interesting, please consider clicking **Like**.  
Likes directly help in the AMD x Hugging Face hackathon public ranking.

## Disclaimer

Demo running on CPU with pre-rendered results. Real-time inference requires AMD MI300X.
