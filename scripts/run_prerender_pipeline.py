"""Run one prerender pipeline job and persist benchmark metadata."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from api.pipeline_runner import run_pipeline_job


def _format_seconds(value: float) -> str:
    seconds = max(0, int(value))
    minutes, rem = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {rem}s"
    if minutes > 0:
        return f"{minutes}m {rem}s"
    return f"{rem}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one full pipeline pass for prerendered data.")
    parser.add_argument("--video-path", required=True, help="Absolute or relative path to source video.")
    parser.add_argument("--video-id", required=True, help="Stable output ID under --data-root.")
    parser.add_argument("--task-id", default=None, help="Optional explicit task ID.")
    parser.add_argument("--data-root", default="data/prerendered", help="Root output directory.")
    parser.add_argument(
        "--status-db",
        default="data/prerendered/_pipeline_status.sqlite3",
        help="SQLite path for multi-video run status tracking.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    status_db = Path(args.status_db).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    status_db.parent.mkdir(parents=True, exist_ok=True)

    task_id = args.task_id or f"prerender-{args.video_id}-{int(time.time())}"
    video_path = str(Path(args.video_path).resolve())
    semantic_started_at = time.perf_counter()
    last_semantic_print_at = 0.0

    def emit(payload: dict[str, Any]) -> None:
        stage = payload.get("stage", "unknown")
        percent = payload.get("percent", 0)
        message = payload.get("message")
        if message:
            print(f"[stage] {stage} {percent}% - {message}")
            return
        print(f"[stage] {stage} {percent}%")

    def semantic_progress(payload: dict[str, Any]) -> None:
        nonlocal last_semantic_print_at
        completed = int(payload.get("completed_crops", 0))
        total = int(payload.get("total_crops", 0))
        eta_sec = float(payload.get("eta_sec", 0.0))
        track_id = int(payload.get("track_id", -1))
        now = time.perf_counter()
        # Keep logs readable for long jobs while still surfacing progress.
        should_print = completed <= 3 or completed == total or (now - last_semantic_print_at) >= 5.0
        if not should_print:
            return
        elapsed = now - semantic_started_at
        print(
            "[semantic] "
            f"{completed}/{total} crops | track={track_id} | "
            f"elapsed={_format_seconds(elapsed)} | eta={_format_seconds(eta_sec)}"
        )
        last_semantic_print_at = now

    started_at = time.perf_counter()
    result = run_pipeline_job(
        video_id=args.video_id,
        video_url=video_path,
        task_id=task_id,
        data_root=data_root,
        status_db_path=status_db,
        emit=emit,
        semantic_progress=semantic_progress,
    )
    elapsed = time.perf_counter() - started_at

    video_dir = data_root / args.video_id
    benchmark_path = video_dir / "benchmarks.json"
    benchmark_payload = {
        "video_id": args.video_id,
        "task_id": task_id,
        "video_path": video_path,
        "finished_at_unix": time.time(),
        "elapsed_wall_sec": round(elapsed, 3),
        "benchmark": result.get("benchmark", {}),
        "vision_summary": result.get("vision_summary", {}),
        "db_path": result.get("db_path"),
        "timeline_path": result.get("timeline_path"),
    }
    benchmark_path.write_text(json.dumps(benchmark_payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "video_id": args.video_id,
                "task_id": task_id,
                "elapsed_wall_sec": round(elapsed, 3),
                "db_path": result.get("db_path"),
                "timeline_path": result.get("timeline_path"),
                "benchmarks_path": str(benchmark_path),
                "benchmark": result.get("benchmark", {}),
                "vision_summary": result.get("vision_summary", {}),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
