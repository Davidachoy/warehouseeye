"""Operation Overview tab rendering."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.services.annotated_video import ensure_annotated_video
from frontend.services.video_registry import VideoRecord


@dataclass
class OverviewStats:
    people_detected: int
    duration_sec: float
    anomalies: int


def _build_track_segments(timeline_rows: list[dict[str, Any]]) -> pd.DataFrame:
    grouped: dict[int, list[float]] = defaultdict(list)
    for entry in timeline_rows:
        track_id = int(entry.get("track_id", -1))
        ts = float(entry.get("timestamp_sec", 0.0))
        grouped[track_id].append(ts)

    rows: list[dict[str, Any]] = []
    origin = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for track_id, timestamps in grouped.items():
        if not timestamps:
            continue
        start_sec = min(timestamps)
        end_sec = max(timestamps) + 1.0
        rows.append(
            {
                "track_label": f"Track {track_id}",
                "track_id": track_id,
                "start_time": origin + timedelta(seconds=start_sec),
                "end_time": origin + timedelta(seconds=end_sec),
                "duration_sec": round(end_sec - start_sec, 1),
            }
        )
    return pd.DataFrame(rows)


def _compute_stats(timeline_rows: list[dict[str, Any]]) -> OverviewStats:
    unique_tracks = {int(entry.get("track_id", -1)) for entry in timeline_rows if entry.get("track_id") is not None}
    duration = 0.0
    anomalies = 0
    for entry in timeline_rows:
        duration = max(duration, float(entry.get("timestamp_sec", 0.0)))
        activity = entry.get("activity", {})
        if isinstance(activity, dict) and bool(activity.get("anomaly")):
            anomalies += 1
    return OverviewStats(
        people_detected=len(unique_tracks),
        duration_sec=duration,
        anomalies=anomalies,
    )


def _render_video_player(video_path: Path) -> None:
    # Keep the player readable on ultrawide displays while remaining fluid on small screens.
    st.markdown(
        """
        <style>
        video.stVideo,
        [data-testid="stVideo"] {
            display: block;
            width: 100%;
            max-width: min(960px, 100%);
            margin-left: auto;
            margin-right: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.video(str(video_path))


def _plot_timeline(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No timeline segments available for this video yet.")
        return
    fig = px.timeline(
        df,
        x_start="start_time",
        x_end="end_time",
        y="track_label",
        color="track_label",
        hover_data={"duration_sec": True, "track_id": True},
    )
    fig.update_layout(
        paper_bgcolor="#0F141A",
        plot_bgcolor="#0F141A",
        legend_title_text="Track",
        font={"family": "JetBrains Mono, Fira Code, monospace", "color": "#E8EEF2"},
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis_title="Timeline Window",
        yaxis_title="Identity Tracks",
    )
    fig.update_traces(marker_line_width=0)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch", theme=None)


def render_operation_overview(video: VideoRecord, timeline_rows: list[dict[str, Any]]) -> None:
    if video.video_path.exists():
        show_overlay = st.toggle("Mostrar IDs sobre el video", value=True, key=f"overlay-{video.video_id}")
        render_path = video.video_path
        if show_overlay:
            with st.spinner("Generando overlay de IDs..."):
                annotated_path = ensure_annotated_video(
                    video_path=video.video_path,
                    video_id=video.video_id,
                    timeline_rows=timeline_rows,
                )
            if annotated_path is not None:
                render_path = annotated_path
            else:
                st.info("No se pudo generar overlay para este video. Mostrando video original.")
        _render_video_player(render_path)
    else:
        st.warning(f"Video file unavailable: {video.video_path}")

    stats = _compute_stats(timeline_rows)
    col1, col2, col3 = st.columns(3)
    col1.metric("People Detected", f"{stats.people_detected}")
    col2.metric("Total Duration", f"{stats.duration_sec:.1f}s")
    col3.metric("Anomalies Detected", f"{stats.anomalies}")

    st.markdown("#### Activity Timeline by Track")
    timeline_df = _build_track_segments(timeline_rows)
    _plot_timeline(timeline_df)
