"""WarehouseEye Streamlit demo frontend."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import streamlit as st

from benchmark.tracker import BenchmarkTracker
from frontend.components.layout import render_footer, render_header, apply_theme
from frontend.components.overview_tab import render_operation_overview
from frontend.components.performance_tab import render_performance_dashboard
from frontend.components.query_tab import render_query_tab
from frontend.services.api_client import ApiClient, BackendUnavailableError
from frontend.services.video_registry import VideoRecord, discover_videos

GITHUB_URL = "https://github.com/[user]/warehouseeye"
PAPER_URL = "https://github.com/[user]/warehouseeye#readme"
DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DEFAULT_TIMEOUT_SEC = float(os.getenv("API_TIMEOUT_SEC", "2.0"))


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.strip()) / 4))


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _data_dir() -> Path:
    return _workspace_root() / "data"


@st.cache_data(ttl=30, show_spinner=False)
def load_videos() -> list[VideoRecord]:
    return discover_videos(_data_dir())


@st.cache_data(ttl=15, show_spinner=False)
def load_local_timeline(video_id: str) -> dict[str, Any]:
    timeline_path = _data_dir() / video_id / "timeline.json"
    if not timeline_path.exists():
        return {"timeline": []}
    return json.loads(timeline_path.read_text(encoding="utf-8"))


def _get_api_client() -> ApiClient:
    return ApiClient(base_url=DEFAULT_API_BASE_URL, timeout_sec=DEFAULT_TIMEOUT_SEC)


def _init_state() -> None:
    if "tracker" not in st.session_state:
        st.session_state.tracker = BenchmarkTracker()
    if "timeline_by_video" not in st.session_state:
        st.session_state.timeline_by_video = {}
    if "benchmark_payload" not in st.session_state:
        st.session_state.benchmark_payload = {}
    if "selected_video_id" not in st.session_state:
        videos = load_videos()
        st.session_state.selected_video_id = videos[0].video_id if videos else None
    if "backend_status" not in st.session_state:
        st.session_state.backend_status = {"status": "unknown"}


def _fetch_backend_health(client: ApiClient) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        health = client.get_health()
        st.session_state.tracker.record_request(
            name="health",
            elapsed_sec=time.perf_counter() - started,
            input_tokens=0,
            output_tokens=0,
            success=True,
            stage="health",
        )
        return health
    except BackendUnavailableError:
        st.session_state.tracker.record_failure(
            name="health",
            elapsed_sec=time.perf_counter() - started,
            stage="health",
        )
        return {"status": "unreachable", "vllm_reachable": False}


def _fetch_timeline(client: ApiClient, video: VideoRecord) -> dict[str, Any]:
    if video.video_id in st.session_state.timeline_by_video:
        return st.session_state.timeline_by_video[video.video_id]

    started = time.perf_counter()
    try:
        payload = client.get_timeline(video.video_id)
        elapsed = time.perf_counter() - started
        st.session_state.tracker.record_request(
            name="timeline",
            elapsed_sec=elapsed,
            input_tokens=0,
            output_tokens=_estimate_tokens(json.dumps(payload)),
            success=True,
            stage="timeline",
        )
    except Exception:
        payload = load_local_timeline(video.video_id)
        elapsed = time.perf_counter() - started
        st.session_state.tracker.record_failure(
            name="timeline",
            elapsed_sec=elapsed,
            stage="timeline",
        )
    st.session_state.timeline_by_video[video.video_id] = payload
    return payload


def _select_video(videos: list[VideoRecord]) -> VideoRecord | None:
    if not videos:
        return None

    options = [video.video_id for video in videos]
    current = st.session_state.selected_video_id
    if current not in options:
        current = options[0]

    index = options.index(current)
    selected_id = st.sidebar.selectbox("Select pre-processed video", options=options, index=index)
    st.session_state.selected_video_id = selected_id
    return next(item for item in videos if item.video_id == selected_id)


def _render_sidebar(video: VideoRecord | None, client: ApiClient) -> None:
    st.sidebar.header("Demo Controls")
    if video is None:
        st.sidebar.warning("No demo videos found in data directory.")
        return

    if st.sidebar.button("Re-analyze", use_container_width=True, type="primary"):
        started = time.perf_counter()
        with st.spinner("Submitting analysis job..."):
            try:
                response = client.analyze(video_id=video.video_id, video_url=str(video.video_path))
                st.sidebar.success(f"Pipeline status: {response.get('status')} (task {response.get('task_id')})")
                st.session_state.tracker.record_request(
                    name="analyze",
                    elapsed_sec=time.perf_counter() - started,
                    input_tokens=0,
                    output_tokens=_estimate_tokens(json.dumps(response)),
                    success=True,
                    stage="analyze",
                )
            except Exception as exc:
                st.session_state.tracker.record_failure(
                    name="analyze",
                    elapsed_sec=time.perf_counter() - started,
                    stage="analyze",
                )
                st.sidebar.error(f"Analyze request failed: {exc}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.markdown(
        "WarehouseEye turns long warehouse camera footage into searchable intelligence for safety and operations."
    )
    st.sidebar.markdown(f"[GitHub]({GITHUB_URL})")
    st.sidebar.markdown(f"[Paper / Blog]({PAPER_URL})")


def _run_query(client: ApiClient, video_id: str, question: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = client.query(video_id=video_id, question=question)
        elapsed = time.perf_counter() - started
        st.session_state.tracker.record_request(
            name="query",
            elapsed_sec=elapsed,
            input_tokens=_estimate_tokens(question),
            output_tokens=_estimate_tokens(response.get("narrative", "")),
            success=True,
            stage="query",
        )
        return response
    except BackendUnavailableError:
        elapsed = time.perf_counter() - started
        st.session_state.tracker.record_failure(name="query", elapsed_sec=elapsed, stage="query")
        return {
            "matched_track_id": None,
            "ambiguous": False,
            "alternatives": [],
            "narrative": "Backend is offline. Start FastAPI to enable natural-language queries.",
            "timeline": [],
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        st.session_state.tracker.record_failure(name="query", elapsed_sec=elapsed, stage="query")
        return {
            "matched_track_id": None,
            "ambiguous": False,
            "alternatives": [],
            "narrative": f"No matching track found for that query. ({exc})",
            "timeline": [],
        }


def _resolve_track(client: ApiClient, video_id: str, track_id: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        payload = client.get_track_timeline(video_id=video_id, track_id=track_id)
        st.session_state.tracker.record_request(
            name="track_refine",
            elapsed_sec=time.perf_counter() - started,
            input_tokens=0,
            output_tokens=_estimate_tokens(json.dumps(payload)),
            success=True,
            stage="query",
        )
        return {
            "matched_track_id": track_id,
            "ambiguous": False,
            "alternatives": [],
            "narrative": f"Displaying events for track {track_id}.",
            "timeline": payload.get("timeline", []),
        }
    except Exception:
        st.session_state.tracker.record_failure(
            name="track_refine",
            elapsed_sec=time.perf_counter() - started,
            stage="query",
        )
        return {
            "matched_track_id": None,
            "ambiguous": False,
            "alternatives": [],
            "narrative": "Unable to refine that candidate right now.",
            "timeline": [],
        }


def _load_benchmark(client: ApiClient, timeline_rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        payload = client.benchmark()
    except Exception:
        payload = st.session_state.benchmark_payload or {}

    defaults = {
        "frames_analyzed": len(timeline_rows),
        "tokens_per_second_avg": 0.0,
        "latency_per_crop_ms": 0.0,
        "wall_time_sec": st.session_state.tracker.wall_time_sec,
        "total_cost_usd": st.session_state.tracker.cumulative_cost_usd,
    }
    merged = {**defaults, **payload}
    st.session_state.benchmark_payload = merged
    return merged


def main() -> None:
    apply_theme()
    _init_state()
    render_header()

    videos = load_videos()
    client = _get_api_client()
    selected_video = _select_video(videos)
    _render_sidebar(selected_video, client)

    health = _fetch_backend_health(client)
    st.session_state.backend_status = health
    if health.get("status") == "unreachable":
        st.error(
            "FastAPI backend is not reachable. Run `uvicorn api.main:app --reload` and refresh this page."
        )
    else:
        vllm_status = "online" if health.get("vllm_reachable") else "degraded"
        st.markdown(
            f'<span class="status-pill">Backend: {health.get("status", "unknown")}</span>'
            f'<span class="status-pill">VLLM: {vllm_status}</span>',
            unsafe_allow_html=True,
        )

    if selected_video is None:
        st.info("Place videos in `data/videos_processed` or `data/videos` to start the demo.")
        render_footer()
        return

    timeline_payload = _fetch_timeline(client, selected_video)
    timeline_rows = timeline_payload.get("timeline", [])

    tab_overview, tab_query, tab_perf = st.tabs(
        ["Operation Overview", "Ask the Video", "Performance Dashboard"]
    )

    with tab_overview:
        render_operation_overview(selected_video, timeline_rows)

    with tab_query:
        render_query_tab(
            video_id=selected_video.video_id,
            run_query=lambda prompt: _run_query(client, selected_video.video_id, prompt),
            resolve_track=lambda track_id: _resolve_track(client, selected_video.video_id, track_id),
            workspace_root=_workspace_root(),
        )

    with tab_perf:
        benchmark_payload = _load_benchmark(client, timeline_rows)
        throughput_rows = st.session_state.tracker.throughput_series()
        if not throughput_rows and benchmark_payload.get("wall_time_sec", 0) > 0:
            throughput_rows = [
                {
                    "elapsed_sec": float(benchmark_payload["wall_time_sec"]),
                    "tokens_per_sec": float(benchmark_payload.get("tokens_per_second_avg", 0.0)),
                }
            ]
        render_performance_dashboard(benchmark_payload, throughput_rows)

    render_footer()


if __name__ == "__main__":
    main()
