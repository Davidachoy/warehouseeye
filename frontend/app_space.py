"""WarehouseEye Streamlit entrypoint for Hugging Face Spaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from frontend.components.overview_tab import render_operation_overview
from frontend.components.performance_tab import render_performance_dashboard
from frontend.components.space_layout import (
    apply_space_theme,
    render_hero,
    render_how_it_works,
    render_space_footer,
)
from frontend.components.space_query_tab import render_space_query_tab
from frontend.services.space_data import (
    SpaceVideoRecord,
    discover_space_videos,
    load_benchmarks,
    load_timeline,
)

GITHUB_URL = "https://github.com/Davidachoy/warehouseeye"
HF_SPACE_URL = "https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/warehouseeye"
TWITTER_URL = "https://x.com/achoy__"
PAPER_URL = ""

PRERENDERED_ROOT = Path(__file__).resolve().parents[1] / "data" / "prerendered"


@st.cache_data(show_spinner=False)
def get_space_videos() -> list[SpaceVideoRecord]:
    return discover_space_videos(PRERENDERED_ROOT)


def _init_state(videos: list[SpaceVideoRecord]) -> None:
    if "space_selected_video_id" not in st.session_state:
        st.session_state.space_selected_video_id = videos[0].video_id if videos else None


def _select_video(videos: list[SpaceVideoRecord]) -> SpaceVideoRecord | None:
    st.sidebar.header("Demo Controls")
    if not videos:
        st.sidebar.warning("No pre-rendered videos found under data/prerendered/.")
        return None

    options = [video.video_id for video in videos]
    current = st.session_state.space_selected_video_id
    if current not in options:
        current = options[0]

    selected_id = st.sidebar.selectbox(
        "Select pre-rendered video",
        options=options,
        index=options.index(current),
    )
    st.session_state.space_selected_video_id = selected_id
    st.sidebar.markdown("---")
    st.sidebar.caption("Backend-free demo: all responses are loaded from pre-rendered files.")
    return next(video for video in videos if video.video_id == selected_id)


def _benchmark_payload(raw: dict[str, Any], timeline_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "frames_analyzed": int(raw.get("frames_analyzed", len(timeline_rows))),
        "tokens_per_second_avg": float(raw.get("tokens_per_second_avg", 0.0)),
        "latency_per_crop_ms": float(raw.get("latency_per_crop_ms", 0.0)),
        "wall_time_sec": float(raw.get("wall_time_sec", 0.0)),
        "total_cost_usd": float(raw.get("total_cost_usd", 0.0)),
    }


def main() -> None:
    apply_space_theme()
    render_hero(github_url=GITHUB_URL, hf_space_url=HF_SPACE_URL)
    render_how_it_works()

    videos = get_space_videos()
    _init_state(videos)
    selected_video = _select_video(videos)

    st.markdown(
        '<span class="badge">Runtime: CPU Basic (2 vCPU / 16 GB)</span>'
        '<span class="badge">Mode: Pre-rendered demo</span>',
        unsafe_allow_html=True,
    )

    if selected_video is None:
        st.info("Add pre-rendered outputs under `data/prerendered/<video_id>/` and refresh.")
        render_space_footer(github_url=GITHUB_URL, paper_url=PAPER_URL, twitter_url=TWITTER_URL)
        return

    timeline_payload = load_timeline(selected_video)
    timeline_rows = timeline_payload.get("timeline", [])
    benchmark_payload = _benchmark_payload(load_benchmarks(selected_video), timeline_rows)

    tab_overview, tab_query, tab_perf = st.tabs(
        ["Operation Overview", "Ask the Video", "Performance Dashboard"]
    )

    with tab_overview:
        render_operation_overview(selected_video, timeline_rows)

    with tab_query:
        render_space_query_tab(
            timeline_rows=timeline_rows,
            workspace_root=Path(__file__).resolve().parents[1],
            github_url=GITHUB_URL,
        )

    with tab_perf:
        throughput_rows = []
        if benchmark_payload["wall_time_sec"] > 0:
            throughput_rows = [
                {
                    "elapsed_sec": benchmark_payload["wall_time_sec"],
                    "tokens_per_sec": benchmark_payload["tokens_per_second_avg"],
                }
            ]
        render_performance_dashboard(benchmark_payload, throughput_rows)

    render_space_footer(github_url=GITHUB_URL, paper_url=PAPER_URL, twitter_url=TWITTER_URL)


if __name__ == "__main__":
    main()
