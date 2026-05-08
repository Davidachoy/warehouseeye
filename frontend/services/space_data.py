"""Prerendered data discovery and loaders for HF Space demo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st


@dataclass(frozen=True)
class SpaceVideoRecord:
    video_id: str
    video_path: Path
    timeline_path: Path
    benchmarks_path: Path
    crops_dir: Path


def discover_space_videos(prerendered_root: Path) -> list[SpaceVideoRecord]:
    """Scan data/prerendered and return valid Space-ready videos."""
    records: list[SpaceVideoRecord] = []
    if not prerendered_root.exists():
        return records

    for video_dir in sorted(path for path in prerendered_root.iterdir() if path.is_dir()):
        timeline_path = video_dir / "timeline.json"
        benchmarks_path = video_dir / "benchmarks.json"
        video_path = video_dir / "videos" / "input_video.mp4"
        crops_dir = video_dir / "crops"
        if not timeline_path.exists() or not video_path.exists():
            continue
        records.append(
            SpaceVideoRecord(
                video_id=video_dir.name,
                video_path=video_path,
                timeline_path=timeline_path,
                benchmarks_path=benchmarks_path,
                crops_dir=crops_dir,
            )
        )
    return records


@st.cache_data(show_spinner=False)
def load_timeline(record: SpaceVideoRecord) -> dict[str, Any]:
    payload = json.loads(record.timeline_path.read_text(encoding="utf-8"))
    if "timeline" not in payload or not isinstance(payload["timeline"], list):
        return {"timeline": []}
    return payload


@st.cache_data(show_spinner=False)
def load_benchmarks(record: SpaceVideoRecord) -> dict[str, Any]:
    if not record.benchmarks_path.exists():
        return {}
    payload = json.loads(record.benchmarks_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "benchmark" in payload and isinstance(payload["benchmark"], dict):
        return payload["benchmark"]
    if isinstance(payload, dict):
        return payload
    return {}
